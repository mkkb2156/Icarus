# CLAUDE.md — Icarus 專案技術規格

Icarus 是基於 Embodied AI 的無人機自主立面作業系統（清洗/巡檢）。強化學習策略在 NVIDIA Isaac Lab 中以 GPU 並行模擬訓練（4096 架虛擬無人機、PPO 演算法），匯出為 ONNX/TensorRT 模型後部署至邊緣計算平台。目前處於 **Phase 1（虛擬環境訓練）**，訓練管線已完成。

兩個部署變體：
- **DJI 商用版**：M350 RTK + Manifold 3 + PSDK 角速率+推力控制（本文件）
- **PX4 開源版**：Pixhawk 6X + Jetson Orin NX + ROS 2 Offboard 模式（見 `claude2.md`）

訓練管線（Isaac Lab 環境、獎勵、課程學習、域隨機化、ONNX 匯出）100% 平台無關，兩個變體完全共用。

---

## 產品架構

### 已實作模塊

```
icarus/
  training/                        ← 核心訓練環境（6 個模塊 + agents/）
    facade_drone_env.py            # DirectRLEnv 子類 — 7/7 必要方法全部實作
    facade_drone_env_cfg.py        # @configclass 環境配置（物理參數、動作縮放、獎勵權重）
    reward.py                      # compute_facade_reward() — 6 目標多目標獎勵
    curriculum.py                  # 4 階段課程學習（@dataclass CurriculumStage）
    domain_randomization.py        # @dataclass DomainRandomizationConfig（質量/風/噴水/感測器）
    __init__.py                    # gym.register("Icarus-FacadeDrone-Direct-v0")
    agents/
      rl_games_ppo_cfg.yaml        # PPO 超參數（rl_games a2c_continuous 格式）

scripts/                           ← 訓練與匯出工具鏈
  train.py                         # 訓練入口：AppLauncher → gym.make → RlGamesVecEnvWrapper → Runner
  play.py                          # 推理/可視化：載入 checkpoint → MLP → 1000 步
  export_onnx.py                   # PyTorch → ONNX（opset 17, dynamic batch, 100 組驗證）
  setup_env.sh                     # 一鍵環境安裝（venv + Isaac Lab 2.3.2 + rl_games）

demo/                              ← Streamlit 投資人展示（無需 GPU）
  app.py                           # 首頁：問題/解決方案/3 項指標卡/三層架構圖
  pages/
    1_training_results.py          # 模擬訓練曲線（Plotly）+ 模型規格 + 驗收標準
    2_system_architecture.py       # 三層架構 + M350 規格 + 角速率+推力比較 + 研究基礎
    3_nl_mission_planner.py        # 自然語言 → 弓字型路徑 → Plotly 3D（功能完整）
  requirements.txt                 # streamlit, plotly, anthropic

models/                            ← .onnx / .engine 模型檔（gitignored）
```

### 已規劃但尚未實作（Phase 2+，定義於 PLAN.md）

```
icarus/core/          ← config.py（系統常數）、types.py（StateVector/ObservationTensor）、logging_utils.py
icarus/psdk/          ← bridge.py（C→Python ctypes）、telemetry.py、flight_controller.py、
                        perception.py、data_alignment.py — DJI PSDK 數據層（5 個檔案）
icarus/px4/           ← offboard_node.py、telemetry_node.py、policy_node.py、
                        safety_monitor.py、data_alignment.py — PX4 ROS 2 節點（見 claude2.md）
icarus/inference/     ← engine.py（TensorRT 引擎載入）、policy.py（obs → action 封裝）
icarus/safety/        ← failsafe.py（邊界檢查）、geofence.py（作業區域）、shadow_test.py（影子測試）
icarus/planning/      ← path_planner.py（弓字型路徑）、multi_drone.py（多機協調）
icarus/reporting/     ← report_generator.py（任務後 3D 巡檢報告）
tests/                ← 6 個測試模塊（test_types, test_data_alignment, test_reward,
                        test_failsafe, test_policy, test_path_planner）
```

相關文件：`PLAN.md`（完整技術規格，繁體中文）、`ROADMAP.md`（5 階段產品路線圖，繁體中文）、`claude2.md`（PX4 開源變體規格，英文）。

---

## 核心技術決策

| 決策項 | 選擇 | 依據 |
|--------|------|------|
| Action Space | `[thrust, ω_roll, ω_pitch, ω_yaw]` 角速率+推力 | Swift (Nature 2023), SimpleFlight (RA-L 2024), RAPTOR 共識 |
| Observation | 23 維，旋轉矩陣(9)取代四元數(4) | SimpleFlight: 旋轉矩陣 sim-to-real 成功率顯著更高 |
| 物理/策略頻率 | 200Hz / 50Hz (decimation=4) | Isaac Lab quadcopter 範例標準 |
| 環境模式 | DirectRLEnv（非 ManagerBasedRLEnv） | 官方 quadcopter 範例模式，直接控制力/力矩 |
| 網路架構 | MLP [256, 128, 64] ELU, ~41K 參數 | TensorRT FP16 推理 <1ms，50Hz 策略預算下有 ~19ms 餘裕 |
| 代理機器人 | Crazyflie USD（Phase 2 換 M350 USD） | Isaac Lab 內建資產，Phase 1 驗證用 |
| 目標牆距 | d=1.5m | PLAN.md 規格 |
| 部署控制 | DJI: PSDK angular-rate + thrust / PX4: Offboard body rate + thrust | 兩者等效，外迴路設定點控制 |
| DJI 安全架構 | DJI 安全仲裁器始終在線（馬達混控、ESC、權限管理、失效保護）| 商業優勢，非技術限制 |

---

## 訓練環境模塊設計

### FacadeDroneEnv (`facade_drone_env.py`)

核心環境類，繼承 Isaac Lab `DirectRLEnv`，實作全部 7 個必要方法：
`_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`

**場景構成**：
- Robot: `Articulation`（Crazyflie USD），初始位置 (1.5, 0, 1.5)
- Wall: `RigidObject`（0.1×50×100m 靜態立方體），位於 x=0
- Ground plane + Dome light

**觀測空間（23 維）**：
```
lin_vel_body(3)        # 機體線速度（世界→機體座標轉換）
ang_vel_body(3)        # 機體角速度
rotation_matrix(9)     # 旋轉矩陣（quat → 3×3 → flatten）
position_error(3)      # 目標位置誤差（當前 - 目標）
distance_to_wall(1)    # 離牆距離（純 x 軸）
wind_estimate(3)       # 風場估計（真實值 + σ=0.1 高斯噪聲）
spray_status(1)        # 噴水狀態（0/1）
```

**動作映射（4 維 → 力/力矩）**：
- `actions[:, 0]` → Z 軸推力：`thrust_to_weight × weight × (a+1)/2`（a=0 時 ≈ 懸停推力）
- `actions[:, 1:4]` → 機體力矩：`moment_scale × actions`（roll/pitch/yaw）

**風場模擬**：Ornstein-Uhlenbeck 過程
- `dv = θ·(0 - v)·dt + σ·√dt·N(0,1)`，θ=0.15，σ=0.3×max_speed
- 風速向量 clamp 至 max_speed，施加為世界座標拖曳力 `F = 0.5·v`

**噴水反衝**：錐形分佈
- 力方向：機體座標 -x（向後），錐半角 0.3 rad (≈17°)
- 力大小：`[0, spray_force_max]` 均勻分佈，乘以 spray_status

**Reset 模式**：
- 調用 `self._robot.reset(env_ids)` + `super()._reset_idx(env_ids)`
- 初始 episode 長度隨機分散（避免 reset 尖峰）
- 位置噪聲 ±init_pos_noise（課程控制）

### 環境配置 (`facade_drone_env_cfg.py`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `sim.dt` | 1/200 (5ms) | 200Hz 物理模擬 |
| `decimation` | 4 | 50Hz 策略頻率 |
| `num_envs` | 4096 | GPU 並行環境數 |
| `episode_length_s` | 30.0 | 每集最長 30 秒 |
| `thrust_to_weight` | 1.9 | 最大推力 = 1.9 × 懸停推力 |
| `moment_scale` | 0.02 Nm | 每單位動作的力矩縮放 |
| `target_wall_distance` | 1.5m | 目標定距 |
| `min_wall_distance` | 0.1m | 碰撞終止閾值 |
| `max_wall_distance` | 5.0m | 漂移終止閾值 |
| `min_altitude` / `max_altitude` | 0.5m / 100m | 高度終止範圍 |
| `rew_alive` | 0.5 | 每 timestep 存活獎勵 |
| `wind_ou_theta` | 0.15 | OU 過程均值回復率 |
| `spray_cone_half_angle` | 0.3 rad | 噴水反衝錐半角 |
| `mass_scale_range` | (0.85, 1.15) | 質量 DR ±15% |

M350 RTK 參考常數（Phase 1 估計值，Phase 2 實測替換）：
- 質量 6.47kg，臂長 0.4475m，最大傾斜 30°
- 最大角速度：pitch/roll 5.24 rad/s (300°/s)，yaw 1.75 rad/s (100°/s)
- 單馬達最大推力 45N [EST]

### 獎勵函數 (`reward.py`)

`compute_facade_reward()` — 6 分量多目標獎勵，所有權重均為 1.0：

| 分量 | 公式 | 設計意圖 |
|------|------|---------|
| 距離維持 | `exp(-2·(d - 1.5)²)` | 高斯型定距獎勵，d=1.5m 時 r=1.0，±1m 時 r≈0.14 |
| 角速度抑制 | `-0.1·Σ(ω²)` | 懲罰機體震盪，促進穩定懸停 |
| 動作平滑度 | `-0.05·Σ(a - a_prev)²` | 懲罰急遽動作變化——SimpleFlight 驗證此為 sim-to-real 關鍵因素 |
| 能量效率 | `-0.01·Σ(a²)` | 減少不必要控制輸出 |
| 存活獎勵 | `+0.5` | 常數項，鼓勵避免碰撞/漂移 |
| 噴水加分 | `spray_status × 1.5 × r_distance` | 噴水時額外精度獎勵（距離維持更重要）|

### 課程學習 (`curriculum.py`)

`@dataclass CurriculumStage` 定義 4 階段遞進式難度：

| 階段 | Epoch 範圍 | 風速上限 | 噴水 | 噴力上限 | 初始位置噪聲 | 描述 |
|------|-----------|---------|------|---------|------------|------|
| 0 | 0–500 | 0 m/s | OFF | 0 N | 0.1m | 靜風懸停 |
| 1 | 500–1500 | 3 m/s | OFF | 0 N | 0.3m | 輕風抗擾 |
| 2 | 1500–3000 | 6 m/s | ON | 8 N | 0.5m | 中風+低壓噴水 |
| 3 | 3000+ | 12 m/s | ON | 15 N | 1.0m | 全域隨機化 |

`get_stage(epoch)` / `get_stage_params(epoch)` 提供查詢介面。

**注意**：階段定義與查詢函數已完成，但與訓練迴圈的整合 hook（在 epoch callback 中自動更新 env cfg）**尚未實作**。

### 域隨機化 (`domain_randomization.py`)

`@dataclass DomainRandomizationConfig` 定義所有 DR 範圍：

**已在環境中啟用**：
- 質量 ±15%（`mass_scale_range = (0.85, 1.15)`，每次 reset 隨機）
- 風場 0–12 m/s（Ornstein-Uhlenbeck 過程，環境內即時模擬）
- 噴水反衝 0–15 N（錐形分佈，半角 0.3 rad）
- 初始位置噪聲 0.1–1.0m（課程控制）

**已定義但尚未套用至環境**：
- 感測器噪聲：IMU 加速度 σ=0.1 m/s²、陀螺儀 σ=0.01 rad/s、深度 σ=0.02m
- 馬達推力係數 ±10%（`thrust_coeff_scale_min/max = 0.90/1.10`）

**刻意跳過**：馬達響應延遲 DR（SimpleFlight 研究顯示影響低）

### PPO 訓練配置 (`rl_games_ppo_cfg.yaml`)

| 類別 | 參數 | 值 |
|------|------|-----|
| 演算法 | name | `a2c_continuous`（rl_games PPO 實作）|
| 網路 | 架構 | MLP [256, 128, 64]，ELU 激活，共享 actor-critic trunk |
| 網路 | 參數量 | ~41K |
| 網路 | sigma | 固定初始化（`fixed_sigma: True`, `val: 0`）|
| PPO | γ (discount) | 0.99 |
| PPO | τ (GAE λ) | 0.95 |
| PPO | Learning rate | 3e-4（adaptive KL，閾值 0.008）|
| PPO | Clip ratio | 0.2 |
| PPO | Entropy coeff | 0.01 |
| PPO | Gradient norm | 1.0（truncate_grads）|
| 批次 | Actors | 4096（= num_envs）|
| 批次 | Minibatch size | 32768 |
| 批次 | Mini epochs | 8 |
| 批次 | Horizon length | 24 |
| 訓練 | Max epochs | 5000 |
| 訓練 | Mixed precision | True（FP16 加速）|
| 訓練 | Normalize input/value | True |
| 存檔 | save_best_after | 100 epochs |
| 存檔 | save_frequency | 500 epochs |

---

## 腳本與工具鏈

- **train.py**：`AppLauncher` → 解析 CLI → `FacadeDroneEnvCfg` → `gym.make("Icarus-FacadeDrone-Direct-v0")` → `RlGamesVecEnvWrapper` → `Runner.run()`。支援 `--checkpoint` 繼續訓練、`--headless` 無渲染模式。
- **play.py**：載入 checkpoint → 建構 MLP → 執行 1000 步推理（wind=12 m/s, spray=15N）。`--record` flag 已定義但視訊錄製功能**未實作**。
- **export_onnx.py**：`build_policy()` → `torch.onnx.export(opset_version=17)`，dynamic_axes 支援任意 batch。100 組隨機輸入驗證 PyTorch vs ONNX 輸出（容差 1e-3）。Jetson 部署需在目標裝置執行 `trtexec --onnx=... --fp16`。
- **setup_env.sh**：先決條件檢查 → Python venv → Isaac Lab 2.3.2 → PyTorch 2.5.1+cu124 → rl_games 1.6.1 → 專案 `pip install -e ".[training]"`。

---

## Demo 應用（Streamlit 投資人展示）

三頁 Streamlit 應用，無需 GPU，用於產品展示與投資人溝通。

**app.py — 首頁**：
- 問題陳述（高空作業風險、效率低、品質不穩定）
- 解決方案概述（Embodied AI、模擬訓練、<1ms 反應時間）
- 3 項關鍵指標卡：4096 並行環境、41K 參數、<1ms 推理
- 三層技術架構圖（虛擬演化 / 數據對齊 / 邊緣執行）

**1_training_results.py — 訓練結果**：
- 模擬訓練曲線（Plotly 折線圖）：平均獎勵、距離誤差、碰撞率
- 模型規格表：架構、參數量、觀測維度、動作維度
- Phase 1 驗收標準（5 項指標，均為 Pending 狀態）
- 影片播放區（`assets/videos/` 目錄，待放入實際訓練影片）

**2_system_architecture.py — 技術架構深入**：
- 三層架構詳解（虛擬演化層 / 數據對齊層 / 邊緣執行層）
- M350 RTK 平台物理規格（質量、尺寸、飛行性能）
- 角速率+推力 vs 速度控制模式比較表
- DJI 安全仲裁器說明（態度 PID 禁用 / 安全仲裁器始終在線）
- 研究基礎表（Swift, SimpleFlight, RAPTOR, Aerial Gym, Isaac Lab）
- 產品路線圖 Phase 0–5 狀態顯示

**3_nl_mission_planner.py — 自然語言任務規劃器**（**功能完整**）：
- 雙語命令解析器：本地 regex 回退 + Claude API tool_use（`claude-haiku-4-5-20251001`，JSON Schema 函數呼叫）
- 弓字型路徑生成器（Boustrophedon pattern）：輸入建築尺寸/樓層/工作區 → 輸出 3D 航點序列
- Plotly 3D 互動式可視化：建築物線框 + 工作區高亮 + 無人機航線 + 起飛/降落標記
- 任務估算：飛行距離(m)、覆蓋面積(m²)、預估時間(min)、電池消耗(%)、用水量(L)
- 支援中英文命令，含 6 個範例（「清洗北面 5-10 樓」、"Clean east facade floors 1-20"）

---

## Isaac Lab DirectRLEnv Pattern

環境嚴格遵循官方 Isaac Lab quadcopter 範例：

- Env class 繼承 `DirectRLEnv`，config 繼承 `DirectRLEnvCfg`
- **7 個必要方法**：`_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`
- `_get_observations()` 回傳 `{"policy": tensor}` dict
- `_get_dones()` 回傳 `(terminated, truncated)` tuple
- 力施加：`self._robot.permanent_wrench_composer.set_forces_and_torques(body_ids=..., forces=..., torques=...)`
- Config 上的 robot 欄位名稱：`robot`（非 `robot_cfg`）
- Body ID 快取：`self._robot.find_bodies("body")[0]`
- Reset 模式：先 `self._robot.reset(env_ids)` 再 `super()._reset_idx(env_ids)`
- Gym 註冊位於 `icarus/training/__init__.py`

---

## 程式碼慣例

- Python >= 3.10；所有模塊以 `from __future__ import annotations` 開頭
- **Ruff** linter，line-length 120（pyproject.toml 配置）
- `@configclass` 用於 Isaac Lab 配置；`@dataclass` 用於非 Isaac Lab 資料結構
- 所有 tensor 運算向量化（N 並行環境）— 禁止 Python 迴圈遍歷環境
- 函數簽名需有型別標註
- 模塊層級 docstring 說明用途與學術引用
- 程式碼與 docstring 使用**英文**；PLAN.md / ROADMAP.md 使用**繁體中文**
- Isaac Lab 必須在專案模塊之前匯入（見 train.py 匯入順序）

---

## 部署變體

| 變體 | 飛行平台 | 飛控 | 運算平台 | 通訊介面 | 安全架構 | 規格文件 |
|------|---------|------|---------|---------|---------|---------|
| DJI 商用 | M350 RTK | DJI 內部 FC（黑盒） | Manifold 3 (Jetson Orin NX 16GB) | PSDK C bridge (ctypes) | DJI 安全仲裁器（始終在線） | 本文件 |
| PX4 開源 | 自組機架 (650-900mm) | Pixhawk 6X (STM32H753) | Jetson Orin NX 16GB | ROS 2 + Micro XRCE-DDS | PX4 failsafe（開源可配置） | `claude2.md` |

**兩個變體共用的模塊**（100% 平台無關）：
- 訓練環境：`icarus/training/`（env, config, reward, curriculum, DR）
- 訓練腳本：`scripts/train.py`, `scripts/play.py`
- 模型匯出：`scripts/export_onnx.py`
- 推理引擎：TensorRT FP16（ONNX → `.engine`）
- MLP 策略：[256, 128, 64] ELU，~41K 參數
- Demo 應用：`demo/`

**DJI 特有模塊**（Phase 2 開發）：`icarus/psdk/`
- PSDK 角速率+推力模式：`HORIZONTAL_ANGULAR_RATE` + `VERTICAL_THRUST` + `YAW_ANGLE_RATE`，`HORIZONTAL_BODY_COORDINATE`
- DJI 安全仲裁器始終在線：馬達混控、ESC 保護、Joystick 權限管理（RC 可隨時收回）、失效保護
- 三項 Phase 2 驗證：(a) 指令速率 & 延遲 (b) 模式可用性 (c) 權限接管邊界

**PX4 特有模塊**（Phase 2 開發）：`icarus/px4/`
- PX4 Offboard 模式：`VehicleRatesSetpoint`（body rate + thrust）
- ROS 2 Humble + Micro XRCE-DDS over Ethernet（~5-10ms 延遲）
- PX4 failsafe 完全開源可配置（`COM_OF_LOSS_T`, `COM_RC_OVERRIDE`）
- 詳見 `claude2.md`

---

## 實作進度

| 模塊 | 狀態 | 說明 |
|------|------|------|
| 訓練環境 (`FacadeDroneEnv`) | ✅ 完成 | 7/7 DirectRLEnv 方法全部實作，風場/噴水模擬運作正常 |
| 環境配置 (`FacadeDroneEnvCfg`) | ✅ 完成 | 所有物理參數、動作縮放、終止條件已配置 |
| 獎勵函數 (`compute_facade_reward`) | ✅ 完成 | 6 個分量全部啟用（距離/震盪/平滑/能量/存活/噴水）|
| 課程學習 (`curriculum.py`) | ⚠️ 部分完成 | 4 階段定義 + 查詢函數完成；訓練迴圈自動整合 hook **未實作** |
| 域隨機化 (`domain_randomization.py`) | ⚠️ 部分完成 | 質量/風/噴水/位置已啟用；感測器噪聲 + 馬達推力係數**已定義未套用** |
| PPO 訓練管線 | ✅ 完成 | rl_games 整合、checkpoint 儲存/載入、mixed precision |
| ONNX 匯出 + 驗證 | ✅ 完成 | opset 17, dynamic batch, 100 組自動驗證 |
| Streamlit Demo (3 頁) | ✅ 完成 | 全功能，使用模擬訓練數據 |
| NL 任務規劃器 | ✅ 完成 | 雙語、雙後端（regex 回退 + Claude API tool_use）|
| M350 USD 資產 | ❌ Phase 2 | 目前使用 Crazyflie 代理 |
| 核心型別系統 (`icarus/core/`) | ❌ Phase 1 待完成 | StateVector, ObservationTensor, FlightCommandInterface |
| PSDK 數據層 (`icarus/psdk/`) | ❌ Phase 2 | C bridge, 遙測, 飛控介面（5 個檔案）|
| PX4 ROS 2 節點 (`icarus/px4/`) | ❌ Phase 2 | 僅規格文件（claude2.md）|
| TensorRT 推理 (`icarus/inference/`) | ❌ Phase 3 | 引擎載入 + 策略封裝 |
| 安全/失效保護 (`icarus/safety/`) | ❌ Phase 4 | failsafe, geofence, shadow test |
| 路徑規劃 (`icarus/planning/`) | ❌ Phase 5 | 弓字型路徑 + 多機協調 |
| 測試模塊 (`tests/`) | ❌ 未開始 | 0/6 已規劃測試模塊 |

---

## 建置與執行

```bash
# 環境安裝（Ubuntu 22.04, Python 3.10, NVIDIA RTX GPU）
./scripts/setup_env.sh

# 快速煙霧測試
python scripts/train.py --num_envs 64 --max_iterations 10

# 完整訓練（RTX 4090 約 2-6 小時）
python scripts/train.py --num_envs 4096 --headless --max_iterations 5000

# 可視化已訓練策略
python scripts/play.py --num_envs 32 --checkpoint logs/latest/nn/best.pth

# 匯出 ONNX
python scripts/export_onnx.py --checkpoint logs/latest/nn/best.pth --output models/facade_policy.onnx

# 投資人 Demo（無需 GPU）
cd demo && pip install -r requirements.txt && streamlit run app.py
```

安裝選項：`pip install -e ".[training]"`（Isaac Lab）、`pip install -e ".[dev]"`（pytest, ruff）、`pip install -e ".[demo]"`（streamlit, plotly, anthropic）。

---

## 術語表

| 術語 | 說明 |
|------|------|
| 角速率+推力 (Angular Rate + Thrust) | PSDK 最低層級搖桿模式：`HORIZONTAL_ANGULAR_RATE` + `VERTICAL_THRUST` + `YAW_ANGLE_RATE`，機體座標 (FRU)。RL 文獻常稱 "CTBR"。DJI 安全仲裁器始終在線——這是高頻外迴路設定點控制，非原始馬達存取。|
| DJI 安全仲裁器 | DJI 始終在線的安全層：Joystick 權限管理（RC 可於暫停/低電量/地理圍欄/PSDK 斷線時收回控制權）、馬達混控、ESC 保護、失效保護觸發。商業合規優勢。|
| PSDK | Payload SDK — DJI 機載電腦 (Manifold 3) 的 API |
| M350 RTK | DJI Matrice 350 RTK — 目標工業無人機平台（6.47kg, 4 旋翼）|
| Manifold 3 | DJI 機載電腦（Jetson Orin NX 16GB），執行 TensorRT FP16 推理 |
| DR (域隨機化) | Domain Randomization — 質量 ±15%、風 0-12 m/s、噴水 0-15 N |
| 課程學習 (Curriculum) | 4 階段訓練難度遞增：靜風 → 輕風 → 中風+噴水 → 全域隨機化 |
| 弓字型路徑 (Boustrophedon) | 立面覆蓋的割草機式往復路徑規劃模式 |
| SimpleFlight | IEEE RA-L 2024 — 旋轉矩陣輸入 + 動作平滑度懲罰是 sim-to-real 關鍵 |
| Swift | Nature 2023 — RL 無人機競速冠軍，驗證角速率+推力 action space |
| DirectRLEnv | Isaac Lab 環境模式，直接控制力/力矩（非 Manager-based 抽象層）|
| PX4 Offboard | PX4 飛行模式，外部電腦提供設定點，等效於 DJI PSDK joystick 控制權。詳見 `claude2.md` |
