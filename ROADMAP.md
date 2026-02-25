# Icarus 產品路線圖 (Product Roadmap)
## 基於 Embodied AI 之無人機立面作業系統

---

## 研究基礎 (Research Foundation)

本路線圖基於以下已驗證的最新技術成果：

| 參考專案 | 來源 | 關鍵啟發 |
|---------|------|---------|
| **Swift** (UZH, Nature 2023) | 冠軍級無人機競速 | 非參數式經驗噪聲模型、CTBR action space |
| **RAPTOR** (rl-tools, 2025) | 2084 參數 GRU 控制 10 種不同無人機 | Meta-Imitation Learning、隱式 System ID |
| **SimpleFlight** (清華, IEEE RA-L 2024) | 零樣本 Sim-to-Real 五大關鍵因素 | 旋轉矩陣輸入、動作平滑懲罰、SysID + 選擇性 DR |
| **Aerial Gym** (NTNU, IEEE RA-L 2025) | GPU 並行數千架 MAV | Isaac Gym 架構模板、深度導航 |
| **OmniDrones** (Duke/清華, IEEE RA-L 2024) | 多旋翼 RL benchmark | MultirotorBase 架構、多代理任務 |
| **AirGym/emNavi** (清華, 2025) | IsaacGym → 真實戶外飛行 | rlPx4Controller、AirGym-Real 部署模組 |
| **Isaac Lab 內建 Quadcopter** (NVIDIA) | 官方 DirectRLEnv 範例 | 完整訓練-部署-匯出管線 |

### 關鍵技術發現（影響架構決策）

1. **Action Space**: CTBR (Collective Thrust + Body Rates) 是 sim-to-real 成功率最高的 action space（Swift, SimpleFlight, RAPTOR 一致結論）。PSDK header `dji_flight_controller.h` 定義了 `HORIZONTAL_ANGULAR_RATE_CONTROL_MODE=3`、`VERTICAL_THRUST_CONTROL_MODE=2`、`STABLE_CONTROL_MODE_DISABLE=1`，表示 CTBR 在 PSDK 層級可用。
   - **Action space**: `[thrust, ω_roll, ω_pitch, ω_yaw]` — 直接 body-rate + thrust 控制，繞過 DJI 內部態度控制器
   - Phase 2 上機後第一優先驗證 M350 是否接受 CTBR 指令；若韌體拒絕再實作 velocity fallback（訓練僅需 ~4hrs）。

2. **觀察空間修正**: SimpleFlight 研究證實：用 **旋轉矩陣 (9 values)** 取代四元數 (4 values) 作為 actor 輸入能顯著提升 sim-to-real 成功率。PLAN.md 中的觀察向量需相應調整。

3. **Domain Randomization 最佳實踐**:
   - 0% DR = sim-to-real 必定失敗
   - 10% DR = 最佳速度/穩健性平衡
   - 30% DR = 最穩健但犧牲性能
   - **SysID + 選擇性 DR >> 純 DR**（不要隨機化可以量測的參數）
   - 馬達時間常數的 DR 影響低，可忽略

4. **推理延遲**: 我們的 MLP `[20]→256→128→64→[4]` (~41K 參數) 在 Orin NX 上 TensorRT FP16 推理延遲約 **0.1-0.5ms**，50Hz (20ms) 預算下有 ~19ms 餘裕。Python 完全可行。

5. **M350 控制層級**: 無法發送直接馬達 RPM 指令。PSDK 最低層級為 body-rate + thrust（CTBR）。PSDK header 定義了 `HORIZONTAL_ANGULAR_RATE_CONTROL_MODE` 及 `VERTICAL_THRUST_CONTROL_MODE` 枚舉，**CTBR 可能可用但需 Phase 2 實機驗證**。若可用，RL 策略可直接輸出 body-rate + thrust（繞過 DJI 內部態度控制器）；若不可用，退回 velocity + yaw rate **外迴路控制器**。

6. **Manifold 3 軟體堆疊**: JetPack 5.1.3 / Ubuntu 20.04 / CUDA 11.4 / TensorRT 8.5。TensorRT 引擎必須在 Manifold 3 上建構。

---

## 階段 0：環境準備與基線驗證 (2-3 週)

### 目標
搭建完整的開發環境，確認 Isaac Lab 可正常運作，並用內建範例跑通一個端到端的訓練-推理循環。

### 具體任務

#### 0.1 開發環境搭建
```
硬體需求：
- 訓練機：RTX 4090 (24GB VRAM) 或更高 × 1 台
- 部署機：DJI Manifold 3 (Jetson Orin NX 16GB)
- 測試無人機：DJI M350 RTK × 1 台

軟體安裝：
- Ubuntu 22.04 (訓練機)
- Python 3.11
- Isaac Sim 5.1 (via pip: pip install isaacsim[all])
- Isaac Lab (git clone + ./isaaclab.sh -i)
- RL framework: rl_games + skrl (./isaaclab.sh -i rl_games && ./isaaclab.sh -i skrl)
- PyTorch 2.x, ONNX, TensorRT (訓練機 + Manifold 3)
```

#### 0.2 跑通 Isaac Lab 內建 Quadcopter 範例
```bash
# 訓練 (headless, 4096 並行環境)
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Quadcopter-Direct-v0 --num_envs 4096 --headless --max_iterations 1000

# 驗收 (帶渲染)
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Quadcopter-Direct-v0 --num_envs 32 \
    --checkpoint logs/rl_games/Isaac-Quadcopter-Direct-v0/<timestamp>/nn/last.pth

# 匯出 ONNX
torch.onnx.export(model, dummy_obs, "baseline_policy.onnx", opset_version=17)
```

#### 0.3 基線性能記錄
| 指標 | 目標值 |
|------|--------|
| 訓練收斂 | < 30 分鐘 (RTX 4090, 4096 envs) |
| 訓練吞吐量 | > 300K steps/sec |
| ONNX 匯出 | 成功匯出並用 onnxruntime 驗證 |
| Quadcopter 懸停 | 目視確認在 play 模式中穩定懸停 |

### 交付物
- [x] 可運作的 Isaac Lab 開發環境
- [x] 內建 Quadcopter 範例訓練至收斂
- [x] baseline_policy.onnx 匯出驗證通過
- [x] 環境搭建文檔 (Wiki/README)

---

## 階段 1：虛擬環境立面作業模擬 (6-8 週) ← **您的首要目標**

### 目標
在 Isaac Lab 中建立「大樓立面清潔」專用的虛擬環境，包含牆面模型、噴水反作用力、風場擾動，訓練出能在 d=1.5m 保持穩定距離的 RL 策略。

### 階段 1 完成標準（驗收條件）
```
✅ 4096 並行無人機在虛擬牆面前 d=1.5m 穩定懸停
✅ 開啟噴水模式後依然保持穩定（位置偏差 < ±0.3m）
✅ 12 m/s 側風下不墜機（成功率 > 95%）
✅ 策略匯出為 ONNX 並通過驗證
✅ 完整的訓練指標 dashboard (TensorBoard)
```

### 1A. 自定義環境：FacadeDroneEnv (2 週)

#### 1A.1 場景搭建
```python
# Isaac Lab DirectRLEnv 子類別
class FacadeDroneEnv(DirectRLEnv):
    """
    虛擬環境：無人機面對垂直牆面，目標距離 d=1.5m

    場景元素：
    - 垂直牆面 (50m × 100m flat surface)
    - Crazyflie/自定義四旋翼 (可在 Phase 2 替換為 M350 模型)
    - 地面平面 (ground truth reference)
    """
```

**場景配置：**
```python
@configclass
class FacadeDroneEnvCfg(DirectRLEnvCfg):
    # MDP 維度
    episode_length_s = 30.0          # 30 秒 episodes
    decimation = 4                    # 200Hz physics / 50Hz policy
    action_space = 4                  # [thrust, omega_roll, omega_pitch, omega_yaw] (CTBR)
    observation_space = 23            # 修正後觀察空間 (見 1A.2)
    state_space = 0

    # 物理引擎
    sim: SimulationCfg = SimulationCfg(
        dt=1/200,                     # 200Hz physics step
        render_interval=4,            # 50Hz rendering = 200/4
    )

    # 並行環境
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,              # 5m 間距 (牆面 + 無人機)
        replicate_physics=True,
    )

    # 任務參數
    target_wall_distance = 1.5        # 目標距離 (m)
    wall_dimensions = (50.0, 100.0)   # 牆面寬 × 高 (m)
```

#### 1A.2 觀察空間 (研究修正版)
```python
# 基於 SimpleFlight 研究成果修正：用旋轉矩陣取代四元數
observation_space = {
    # 狀態向量 S (18 dims)
    "linear_velocity_body":   3,    # 體座標系線速度 (m/s)
    "angular_velocity_body":  3,    # 體座標系角速度 (rad/s)
    "rotation_matrix":        9,    # 旋轉矩陣 (比四元數更好的 sim-to-real 效果)
    "position_error":         3,    # 相對於目標位置的誤差

    # 環境向量 E (4 dims)
    "distance_to_wall":       1,    # 毫米波雷達/深度感測 (m)
    "wind_estimate":          3,    # 風速估計向量 (m/s)

    # 任務向量 T (1 dim)
    "spray_status":           1,    # 噴水開關 (0/1)

    # 總計: 23 dims
}
```

#### 1A.3 動作空間 (CTBR)
```python
# CTBR — 對應 PSDK body-rate + thrust 模式
# 所有研究 (Swift, SimpleFlight, RAPTOR) 一致認為 CTBR 是最佳 sim-to-real action space
action_space = {
    "thrust":      (0.0, 1.0),     # 集合推力 (歸一化 0-100%)
    "omega_roll":  (-2.6, 2.6),    # 滾轉角速率 (rad/s, ~150 deg/s)
    "omega_pitch": (-2.6, 2.6),    # 俯仰角速率 (rad/s, ~150 deg/s)
    "omega_yaw":   (-1.7, 1.7),    # 偏航角速率 (rad/s, ~100 deg/s)
}
```

#### 1A.4 必要方法實作
```python
class FacadeDroneEnv(DirectRLEnv):
    def _setup_scene(self):
        """生成牆面 + 無人機 + 地面"""
        # 1. 生成垂直牆面 (cuboid, thin in x, wide in y/z)
        # 2. 生成四旋翼資產 (初始用 Crazyflie, 後替換)
        # 3. 設置地面平面

    def _pre_physics_step(self, actions):
        """將 4D CTBR 動作轉換為力/力矩"""
        # actions = [thrust, omega_roll, omega_pitch, omega_yaw]
        # 直接映射為推力 + 力矩（無需模擬 DJI 內部控制器）

    def _apply_action(self):
        """施加外力（推力 + 噴水反作用力 + 風力）"""
        total_force = self._thrust + self._spray_force + self._wind_force
        self._robot.set_external_force_and_torque(...)

    def _get_observations(self):
        """構建 23 維觀察向量"""
        # 讀取 root_lin_vel_b, root_ang_vel_b, root_rot_w
        # 計算 distance_to_wall (robot_x - wall_x)
        # 讀取 wind_estimate, spray_status

    def _get_rewards(self):
        """計算多目標獎勵"""
        return compute_facade_reward(...)

    def _get_dones(self):
        """終止條件"""
        crashed = distance_to_wall < 0.1    # 撞牆
        drifted = distance_to_wall > 5.0    # 飄走
        timeout = episode_length >= max_len
        return (crashed | drifted), timeout

    def _reset_idx(self, env_ids):
        """重置環境，隨機初始位置"""
        # 初始位置：牆前 1.5m ± 0.5m
        # 初始速度：0 ± 0.2 m/s
```

### 1B. 獎勵函數設計 (1 週)

```python
def compute_facade_reward(
    distance_to_wall: torch.Tensor,      # (N,)
    target_distance: float,               # 1.5m
    angular_velocity: torch.Tensor,       # (N, 3)
    action: torch.Tensor,                 # (N, 4)
    prev_action: torch.Tensor,            # (N, 4)
    spray_status: torch.Tensor,           # (N,) binary
) -> torch.Tensor:
    """
    多目標獎勵函數 — 基於 SimpleFlight 研究的最佳實踐
    """
    # 1. 距離保持 (核心目標)
    distance_error = torch.abs(distance_to_wall - target_distance)
    r_distance = torch.exp(-2.0 * distance_error ** 2)
    # 高斯函數：在 d=1.5m 時 reward=1.0, d=0.5m 或 d=2.5m 時 reward≈0.14

    # 2. 姿態穩定 (抑制震盪)
    r_oscillation = -0.1 * torch.sum(angular_velocity ** 2, dim=-1)

    # 3. 動作平滑 (SimpleFlight 關鍵因素 #3)
    r_smoothness = -0.05 * torch.sum((action - prev_action) ** 2, dim=-1)

    # 4. 能量效率
    r_energy = -0.01 * torch.sum(action ** 2, dim=-1)

    # 5. 存活獎勵
    r_alive = 0.5

    # 6. 噴水補償獎勵 (噴水時保持穩定額外加分)
    r_spray = spray_status * 1.5 * r_distance

    # 加權總和
    total = (1.0 * r_distance
           + 1.0 * r_oscillation
           + 1.0 * r_smoothness
           + 1.0 * r_energy
           + r_alive
           + 1.0 * r_spray)

    return total
```

### 1C. Domain Randomization (1 週)

使用 Isaac Lab 的 `EventManager` API：

```python
@configclass
class FacadeDroneEventCfg:
    """域隨機化配置 — 基於研究最佳實踐"""

    # === 每 Episode 重置隨機化 (mode="reset") ===

    # 質量隨機化 (+/- 15%, 模擬水箱液位變化)
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # 慣量隨機化 (+/- 10%)
    # (透過 mass 的 recompute_inertia=True 已部分處理)

    # === 週期性擾動 (mode="interval") ===

    # 風場推力 (每 3-8 秒隨機變化)
    random_wind = EventTerm(
        func=apply_ornstein_uhlenbeck_wind,  # 自定義函數
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={
            "wind_speed_range": (0.0, 12.0),   # 0-12 m/s
            "theta": 0.15,                       # OU 均值回歸速率
        },
    )

    # 噴水反作用力 (當 spray_status=1)
    spray_recoil = EventTerm(
        func=apply_spray_recoil_force,  # 自定義函數
        mode="interval",
        interval_range_s=(0.02, 0.02),  # 每個 step
        params={
            "force_range": (0.0, 15.0),   # 0-15 N
            "cone_half_angle": 0.3,        # ~17° 向後錐體
        },
    )
```

**不隨機化的參數**（基於 SimpleFlight 研究）：
- 馬達時間常數 → 影響低，不值得
- 過寬的質量範圍 → 浪費訓練算力在不切實際的組合上

### 1D. 課程學習 (Curriculum) (1 週)

```python
class FacadeCurriculum:
    """漸進式難度排程"""

    stages = {
        # Stage 0: 基礎懸停 (0-500 epochs)
        0: {
            "wind_max": 0.0,
            "spray_enabled": False,
            "init_pos_noise": 0.1,     # ±0.1m
            "description": "靜止空氣中懸停於牆面前"
        },
        # Stage 1: 輕風 (500-1500 epochs)
        1: {
            "wind_max": 3.0,           # 0-3 m/s
            "spray_enabled": False,
            "init_pos_noise": 0.3,
            "description": "微風環境下穩定保持距離"
        },
        # Stage 2: 中風 + 噴水 (1500-3000 epochs)
        2: {
            "wind_max": 6.0,           # 0-6 m/s
            "spray_enabled": True,
            "spray_force_max": 8.0,    # 低噴壓
            "init_pos_noise": 0.5,
            "description": "中等風速 + 低壓噴水"
        },
        # Stage 3: 全隨機化 (3000+ epochs)
        3: {
            "wind_max": 12.0,          # 0-12 m/s
            "spray_enabled": True,
            "spray_force_max": 15.0,   # 高壓噴水
            "init_pos_noise": 1.0,
            "description": "全域隨機化：強風 + 高壓噴水"
        },
    }

    def get_stage(self, epoch: int) -> int:
        if epoch < 500: return 0
        if epoch < 1500: return 1
        if epoch < 3000: return 2
        return 3
```

### 1E. 訓練配置與執行 (1 週)

```yaml
# rl_games PPO 配置
params:
  algo:
    name: a2c_continuous
  network:
    mlp:
      units: [256, 128, 64]
      activation: elu
  config:
    num_actors: 4096
    max_epochs: 5000
    minibatch_size: 32768
    mini_epochs: 8
    gamma: 0.99
    tau: 0.95                    # GAE lambda
    learning_rate: 3e-4
    lr_schedule: adaptive
    kl_threshold: 0.008
    e_clip: 0.2                  # PPO clip
    entropy_coef: 0.01
    normalize_input: True
    normalize_value: True
```

**預估訓練時間：**
| GPU | 環境數 | 預估 FPS | 5000 epochs 時間 |
|-----|--------|----------|-----------------|
| RTX 4090 | 4096 | ~400K steps/sec | **2-6 小時** |
| A100 | 4096 | ~300K steps/sec | 3-8 小時 |
| RTX 3080 | 2048 | ~150K steps/sec | 8-16 小時 |

### 1F. 模型匯出與驗證 (1 週)

```python
# Step 1: PyTorch → ONNX
torch.onnx.export(
    policy_net,
    torch.randn(1, 23),
    "facade_policy.onnx",
    opset_version=17,
    input_names=["observation"],
    output_names=["action"],
)

# Step 2: ONNX 簡化
# onnxsim facade_policy.onnx facade_policy_sim.onnx

# Step 3: TensorRT 轉換 (在 Manifold 3 上執行)
# trtexec --onnx=facade_policy_sim.onnx --saveEngine=facade_policy_fp16.engine --fp16

# Step 4: 交叉驗證 (PyTorch vs ONNX vs TensorRT)
# 100 筆隨機輸入，max absolute difference < 1e-3
```

### 階段 1 交付物
| 交付物 | 描述 |
|-------|------|
| `training/facade_drone_env.py` | Isaac Lab 立面清潔環境 |
| `training/reward.py` | 多目標獎勵函數 |
| `training/domain_randomization.py` | DR 配置 (EventManager) |
| `training/curriculum.py` | 4 階段課程學習 |
| `training/facade_ppo_cfg.yaml` | PPO 訓練超參數 |
| `models/facade_policy.onnx` | 匯出的策略模型 |
| TensorBoard 訓練日誌 | reward curve, distance metrics |
| 驗證影片 | Isaac Lab play 模式中的穩定飛行錄影 |

---

## 階段 2：PSDK 整合與 Manifold 3 推理 (4-6 週)

### 目標
在 Manifold 3 上建立完整的「感測 → 推理 → 控制」資料管線，用模擬數據驗證整個軟體堆疊的延遲和穩定性。

### 驗收條件
```
✅ PSDK 成功訂閱 M350 遙測數據 (IMU, GPS, attitude)
✅ TensorRT FP16 推理延遲 < 1ms (trtexec benchmark)
✅ 端到端管線延遲 < 5ms (觀察構建 + 推理 + safety check)
✅ CTBR 指令成功發送 (地面靜態測試，確認 M350 接受 body-rate + thrust 模式)
✅ 50-200Hz CTBR 控制迴路在 Manifold 3 上穩定運行 > 30 分鐘
```

### 2A. PSDK ctypes 封裝 (2 週)
```
psdk/bridge.py          — C → Python 函數封裝
psdk/telemetry.py       — 非同步遙測訂閱 (400Hz angular rate, 200Hz accel/attitude, 50Hz GPS)
psdk/perception.py      — 毫米波雷達深度訂閱
psdk/flight_controller.py — CTBR 指令介面 (body-rate + thrust)
```

**🔴 Phase 2 第一優先驗證：CTBR 模式可用性**
```
在 Manifold 3 上執行：
1. DjiFlightController_SetJoystickMode() 設定 CTBR 枚舉值
   - horizontalControlMode = ANGULAR_RATE (3)
   - verticalControlMode = THRUST (2)
   - stableMode = DISABLE (1)
2. 檢查回傳碼是否為 DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS
3. 若成功 → 確認 CTBR 可用，繼續開發
4. 若失敗 → 記錄錯誤碼，實作 velocity fallback（訓練 ~4hrs）
```

### 2B. 多頻率感測器對齊 (1 週)
```
psdk/data_alignment.py  — 400Hz angular rate + 200Hz accel/attitude + 50Hz GPS + 10Hz Radar → 50-200Hz 觀察 Tensor
策略：IMU 為主時鐘，低頻數據最近鄰插值 + 過期標記
```

### 2C. TensorRT 推理引擎 (1 週)
```
inference/engine.py     — TensorRT 引擎載入器 (零拷貝映射記憶體)
inference/policy.py     — 策略介面 (TensorRT / PyTorch / Dummy 後端)
```

### 2D. 安全攔截層 (1 週)
```
safety/failsafe.py      — 狀態機: NORMAL → CAUTION → OVERRIDE → EMERGENCY
safety/geofence.py      — 3D 操作邊界框
```

### 2E. 控制迴路 & 地面測試 (1 週)
```
controller.py           — 50-200Hz CTBR 主迴路 (觀察→推理→安全→CTBR指令)
main.py                 — 進入點 (CLI, 信號處理, 優雅關機)
目標：Manifold 3 上跑通完整 CTBR 管線，AI 輸出驗證合理性
```

---

## 階段 3：Sim-to-Real 遷移 (4-6 週)

### 目標
收集真實飛行數據進行 System ID，校準模擬器參數，通過影子測試驗證 AI 策略在真實環境中的表現。

### 驗收條件
```
✅ System ID 完成：質量、慣量、推力係數、阻力係數校準
✅ 模擬器與真實飛行軌跡對比誤差 < 15%
✅ 影子測試模式運行 > 5 小時無崩潰
✅ AI 指令與人類飛手指令的偏差分析報告
✅ 基於校準數據重新訓練的 v2 策略
```

### 3A. System Identification (2 週)
```
資料收集（基於 Bauersfeld et al. 2024 方法）：
1. 手動飛行 3 個簡單機動動作（~1 分鐘）
2. 記錄 IMU + 馬達指令 (PSDK ULog)
3. 頻域掃描：每軸注入正弦波（可選，提升精度）

提取參數：
- 質量 (秤重 + 飛行驗證)
- 慣量矩陣 (Ixx, Iyy, Izz) — 資料驅動估算
- 推力係數 (kf) — 懸停推力 / 重力
- 力矩/阻力係數 (kM)
- 重心偏移
- 氣動阻力係數 (高速飛行時)
```

### 3B. 模擬器校準 (1 週)
```
用 SysID 結果更新 Isaac Lab 環境參數
以 SysID 值為中心，DR 範圍縮小至 ±10%
重新訓練 v2 策略 (使用校準後參數)
```

### 3C. 影子測試 (2 週)
```
safety/shadow_test.py:
1. 人類飛手手動控制 M350 進行立面作業
2. AI 在背景運算指令（不執行）
3. 記錄：(timestamp, human_cmd, ai_cmd, observation)
4. 離線分析：計算偏差指標
5. 若偏差 > 閾值 → 調整獎勵函數 → 重新訓練
```

---

## 階段 4：自主飛行驗證 (4-6 週)

### 目標
AI 接管控制權，在受控環境中完成自主立面保持任務。

### 驗收條件
```
✅ AI 自主維持 d=1.5m 牆面距離 > 5 分鐘
✅ 噴水模式下位置偏差 < ±0.5m
✅ 安全系統正確觸發 0 次意外碰撞
✅ 3 級以上側風下穩定飛行
✅ 全程遙測數據完整記錄
```

### 4A. 漸進式授權
```
Level 0: 影子模式（AI 計算，人控飛行）
Level 1: AI 控制 + 人類隨時接管（Virtual Stick authority）
Level 2: AI 全自主 + 安全系統監控
Level 3: AI 全自主 + 路徑規劃
```

### 4B. 安全流程
```
每次飛行前：
1. 地面靜態自檢 (感測器、推理、通訊)
2. 低空懸停測試 (AI 控制 30 秒)
3. 逐漸接近牆面 (2m → 1.5m)
4. 開啟噴水 (低壓 → 高壓)
5. 任何異常 → 立即切回 DJI 原生控制
```

---

## 階段 5：產品化 (8-12 週)

### 5A. 路徑規劃
```
planning/path_planner.py — 弓字型 (Boustrophedon) 清潔路徑生成
planning/multi_drone.py  — 多機協調 (UDP/ROS2)
```

### 5B. 報告系統
```
reporting/report_generator.py — 作業後 3D 清潔覆蓋率報告
```

### 5C. 平台移植
```
MAVLink/PX4 FlightCommandInterface 實作
驗證 AI 模型在非 DJI 平台的可移植性
```

---

## 技術風險登記簿

| 風險 | 影響 | 緩解措施 |
|------|------|---------|
| PSDK Virtual Stick 延遲 > 50ms | 控制頻率受限 | 測量實際延遲，調整策略步進頻率 |
| DJI 內部態度控制器干擾 AI 指令 | 控制品質下降 | 策略在訓練中學習與內部控制器共存 |
| 噴水反作用力非線性超出訓練範圍 | 失控 | DR 範圍覆蓋 20N；安全層硬限制 |
| 大樓表面複雜幾何導致深度感測異常 | 距離保持失敗 | 多感測器融合 (radar + stereo)；過期標記 |
| Manifold 3 熱節流 (不太可能) | 推理延遲增加 | MLP 推理 < 1W，遠低於散熱能力 |
| Isaac Lab 版本升級破壞相容性 | 環境需重寫 | Pin 特定版本；使用 DirectRLEnv (較穩定) |

---

## 關鍵指標追蹤

### 訓練指標
- `reward/mean` — 平均獎勵 (目標: 持續上升)
- `distance_error/mean` — 平均距離偏差 (目標: < 0.1m)
- `distance_error/std` — 距離偏差標準差 (目標: < 0.2m)
- `angular_velocity/norm` — 角速度範數 (目標: < 0.3 rad/s)
- `episode/crash_rate` — 墜機率 (目標: < 1%)
- `action/smoothness` — 動作平滑度 (目標: Δaction_norm 持續下降)

### 部署指標
- `inference_latency/p99` — 推理延遲 P99 (目標: < 2ms)
- `control_loop/frequency` — 實際控制頻率 (目標: 50Hz ± 1Hz)
- `sensor/staleness_rate` — 感測器過期率 (目標: < 0.1%)
- `safety/override_count` — 安全覆寫次數 (目標: 遞減趨勢)
