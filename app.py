# app.py
"""
SharkGuard Streamlit app
- Place this file in the project root (same level as 'sharkguard/' and 'utils/' folders).
- Ensure you have a Python package 'sharkguard' (create an empty sharkguard/__init__.py if needed).
- Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import traceback
import subprocess
import sys
import streamlit.components.v1 as components

# Paths
MODEL_PATH = Path("models/isolation_model.joblib")
SIM_FEATURES = Path("data/simulated_features.csv")
MODELS_DIR = MODEL_PATH.parent
DATA_DIR = SIM_FEATURES.parent

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Attempt to import required project modules; show an informative error if they are missing.
try:
    from fakeacc.core import SharkGuardModel, txs_to_dataframe, extract_wallet_features, train_and_persist_model
except Exception as e:
    st.set_page_config(page_title="SharkGuard - Missing modules", layout="centered")
    st.title("🦈 SharkGuard — Missing project files")
    st.error(
        "Could not import `sharkguard.core`. Make sure:\n"
        "  • `sharkguard/core.py` exists, AND\n"
        "  • `sharkguard/__init__.py` exists (an empty file is fine) so Python treats it as a package.\n\n"
        "Error details:\n" + repr(e)
    )
    st.stop()

# Try to import Etherscan util. If missing we provide a simple stub (app will continue but fetch won't work).
try:
    from utils.etherscan import fetch_transactions, fetch_account_balance
except Exception:
    def fetch_transactions(address, api_key, *args, **kwargs):
        # fallback: return empty list so UI still runs
        return []
    def fetch_account_balance(address, api_key):
        return 0.0

# Import heuristic analysis
try:
    from fakeacc.heuristics import analyze_wallet_heuristics
except Exception:
    def analyze_wallet_heuristics(features, df=None):
        return {'risk_score': 0.0, 'risk_level': 'UNKNOWN', 'flags': [], 'explanations': ['Heuristic analysis unavailable'], 'recommendations': []}

# Ensure a trained model exists and load it. Generates simulated data if needed.
def ensure_model_ready():
    """Ensure the IsolationForest model is trained and available, then load it.
    Returns a loaded SharkGuardModel instance or None if failed."""
    try:
        if not MODEL_PATH.exists():
            # Generate simulated features CSV if missing
            if not SIM_FEATURES.exists():
                with st.spinner("Generating simulated training data..."):
                    # Run the simulator script to produce data/simulated_features.csv
                    subprocess.run([sys.executable, "data/simulate.py"], check=True)
            # Train and persist the model
            with st.spinner("Training built-in model (one-time)..."):
                df = pd.read_csv(SIM_FEATURES)
                train_and_persist_model(df, path=str(MODEL_PATH))
        # Load the model
        return load_model(str(MODEL_PATH))
    except Exception as e:
        st.sidebar.error("Automatic model setup failed: " + str(e))
        st.sidebar.caption("You can still use demo mode without a model.")
        return None

# Cache model loading so Streamlit doesn't reload it every rerun
@st.cache_resource
def load_model(path: str):
    sg = SharkGuardModel()
    sg.load(path)
    return sg

def create_demo_features(wallet_address):
    """
    Create realistic demo features based on wallet address pattern.
    This provides more dynamic and realistic simulations for demonstration.
    """
    import hashlib
    import random
    import time
    
    # Use wallet address + current time for more dynamic results
    current_hour = int(time.time() // 3600)  # Changes every hour
    seed = int(hashlib.md5(f"{wallet_address}{current_hour}".encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    # Different wallet types based on address pattern
    if wallet_address.lower().startswith('0xd8da6bf26964af9d7eed9e03e53415d37aa96045'):
        # Vitalik's address - normal user pattern
        return {
            "tx_count": random.randint(50, 200),
            "tx_freq_per_day": random.uniform(0.5, 2.0),
            "lifetime_days": random.uniform(1000, 2000),
            "avg_gas": random.uniform(50000, 150000),
            "avg_value_eth": random.uniform(0.1, 10.0),
            "unique_counterparties": random.randint(20, 100),
            "repeated_ratio": random.uniform(0.1, 0.3),
            "gas_efficiency": random.uniform(0.1, 2.0),
            "value_variance": random.uniform(0.01, 1.0),
            "weekend_activity": random.uniform(0.1, 0.3),
            "failed_tx_ratio": random.uniform(0.0, 0.05),
            "contract_interaction_ratio": random.uniform(0.1, 0.4)
        }
    elif wallet_address.lower().startswith('0x742d35cc6634c0532925a3b8d4c9db96c4b4d8b6'):
        # Exchange wallet - suspicious pattern
        return {
            "tx_count": random.randint(1000, 5000),
            "tx_freq_per_day": random.uniform(10, 50),
            "lifetime_days": random.uniform(500, 1500),
            "avg_gas": random.uniform(21000, 100000),
            "avg_value_eth": random.uniform(1.0, 100.0),
            "unique_counterparties": random.randint(500, 2000),
            "repeated_ratio": random.uniform(0.05, 0.15),
            "hour_entropy": random.uniform(2.0, 3.5),
            "gas_efficiency": random.uniform(0.5, 5.0),
            "value_variance": random.uniform(10.0, 100.0),
            "weekend_activity": random.uniform(0.3, 0.5),
            "failed_tx_ratio": random.uniform(0.01, 0.05),
            "contract_interaction_ratio": random.uniform(0.6, 0.8)
        }
    elif wallet_address.lower().startswith('0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae'):
        # Foundation wallet - normal organizational pattern
        return {
            "tx_count": random.randint(100, 500),
            "tx_freq_per_day": random.uniform(1.0, 5.0),
            "lifetime_days": random.uniform(2000, 3000),
            "avg_gas": random.uniform(100000, 300000),
            "avg_value_eth": random.uniform(10.0, 1000.0),
            "unique_counterparties": random.randint(50, 200),
            "repeated_ratio": random.uniform(0.2, 0.4),
            "hour_entropy": random.uniform(3.5, 4.8),
            "gas_efficiency": random.uniform(1.0, 10.0),
            "value_variance": random.uniform(100.0, 10000.0),
            "weekend_activity": random.uniform(0.1, 0.3),
            "failed_tx_ratio": random.uniform(0.0, 0.02),
            "contract_interaction_ratio": random.uniform(0.3, 0.6)
        }
    else:
        # Generic wallet - random pattern
        wallet_type = seed % 4
        if wallet_type == 0:  # Normal user
            return {
                "tx_count": random.randint(10, 100),
                "tx_freq_per_day": random.uniform(0.1, 1.0),
                "lifetime_days": random.uniform(100, 1000),
                "avg_gas": random.uniform(50000, 200000),
                "avg_value_eth": random.uniform(0.01, 1.0),
                "unique_counterparties": random.randint(5, 50),
                "repeated_ratio": random.uniform(0.2, 0.6),
                "hour_entropy": random.uniform(3.0, 4.5),
                "gas_efficiency": random.uniform(0.1, 2.0),
                "value_variance": random.uniform(0.01, 1.0),
                "weekend_activity": random.uniform(0.1, 0.3),
                "failed_tx_ratio": random.uniform(0.0, 0.05),
                "contract_interaction_ratio": random.uniform(0.1, 0.4)
            }
        elif wallet_type == 1:  # Bot/suspicious
            return {
                "tx_count": random.randint(500, 2000),
                "tx_freq_per_day": random.uniform(5, 20),
                "lifetime_days": random.uniform(200, 800),
                "avg_gas": random.uniform(21000, 100000),
                "avg_value_eth": random.uniform(0.001, 0.1),
                "unique_counterparties": random.randint(100, 500),
                "repeated_ratio": random.uniform(0.05, 0.2),
                "hour_entropy": random.uniform(1.0, 3.0),
                "gas_efficiency": random.uniform(0.001, 0.1),
                "value_variance": random.uniform(0.0001, 0.01),
                "weekend_activity": random.uniform(0.4, 0.6),
                "failed_tx_ratio": random.uniform(0.05, 0.2),
                "contract_interaction_ratio": random.uniform(0.7, 0.9)
            }
        elif wallet_type == 2:  # New wallet
            return {
                "tx_count": random.randint(1, 10),
                "tx_freq_per_day": random.uniform(0.1, 0.5),
                "lifetime_days": random.uniform(1, 30),
                "avg_gas": random.uniform(21000, 100000),
                "avg_value_eth": random.uniform(0.01, 0.5),
                "unique_counterparties": random.randint(1, 5),
                "repeated_ratio": random.uniform(0.3, 0.8),
                "hour_entropy": random.uniform(2.0, 4.0),
                "gas_efficiency": random.uniform(0.1, 1.0),
                "value_variance": random.uniform(0.01, 0.5),
                "weekend_activity": random.uniform(0.1, 0.4),
                "failed_tx_ratio": random.uniform(0.0, 0.1),
                "contract_interaction_ratio": random.uniform(0.0, 0.3)
            }
        else:  # Dormant wallet
            return {
                "tx_count": random.randint(1, 5),
                "tx_freq_per_day": random.uniform(0.01, 0.1),
                "lifetime_days": random.uniform(1000, 2000),
                "avg_gas": random.uniform(21000, 100000),
                "avg_value_eth": random.uniform(0.1, 5.0),
                "unique_counterparties": random.randint(1, 10),
                "repeated_ratio": random.uniform(0.5, 0.9),
                "hour_entropy": random.uniform(1.0, 2.5),
                "gas_efficiency": random.uniform(0.5, 2.0),
                "value_variance": random.uniform(0.1, 2.0),
                "weekend_activity": random.uniform(0.0, 0.2),
                "failed_tx_ratio": random.uniform(0.0, 0.05),
                "contract_interaction_ratio": random.uniform(0.0, 0.2)
            }

# UI
st.set_page_config(page_title="SharkGuard", layout="wide")
st.title("🦈 SharkGuard — Web3 Fake Account Detector")
st.caption("Detect suspicious or fake Web3 accounts using on‑chain behavioral patterns and anomaly detection.")

# Global styles
st.markdown(
    """
    <style>
      /* Card styling */
      .sg-card {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 12px;
      }
      .sg-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 12px;
        color: #fff;
      }
      .sg-badge.green { background: #16a34a; }
      .sg-badge.orange { background: #f59e0b; }
      .sg-badge.red { background: #dc2626; }
      .sg-badge.gray { background: #6b7280; }
      .sg-riskbar {
        height: 14px; border-radius: 10px; background: #f1f5f9; overflow: hidden;
      }
      .sg-riskbar > div {
        height: 100%;
        background: linear-gradient(90deg, #22c55e 0%, #f59e0b 60%, #ef4444 100%);
      }
      .sg-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      /* Make Streamlit app background transparent so the 3D canvas shows through */
      .stApp { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3D Background Renderer (optional)
def render_3d_background(enable=True, low_power=False, particle_count=1200, speed=1.0,
                         particle_hex="#6aa7ff", bg1="#0b1220", bg2="#060a13", bg3="#03060c"):
    """Inject a full-screen background. If low_power, render static gradient. Otherwise render Three.js particles.

    Args:
        enable (bool): Whether to render any background at all.
        low_power (bool): Static gradient only when True.
        particle_count (int): Number of particles.
        speed (float): Animation speed multiplier.
        particle_hex (str): Particle color, e.g., '#6aa7ff'.
        bg1, bg2, bg3 (str): Gradient colors.
    """
    if not enable:
        return
    # If low power, only render a CSS gradient div
    if low_power:
        components.html(
            f"""
            <div id=\"sg-bg-root\"></div>
            <style>
              #sg-bg-root {{
                position: fixed; inset: 0; z-index: -1; pointer-events: none;
                background: radial-gradient(1200px 600px at 10% 10%, {bg1} 0%, {bg2} 40%, {bg3} 100%);
              }}
            </style>
            """,
            height=0,
            scrolling=False,
        )
        return

    # Full three.js renderer
    particle_color_js = "0x" + particle_hex.replace('#','')
    components.html(
        f"""
        <div id=\"sg-bg-root\"></div>
        <style>
          #sg-bg-root, #sg-bg-canvas {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            width: 100vw; height: 100vh;
            z-index: -1;  /* behind Streamlit content */
            pointer-events: none; /* non-interactive */
          }}
          #sg-bg-root {{ background: radial-gradient(1200px 600px at 10% 10%, {bg1} 0%, {bg2} 40%, {bg3} 100%); }}
        </style>
        <script src=\"https://unpkg.com/three@0.157.0/build/three.min.js\"></script>
        <script>
          (function(){{
            try {{
              const container = document.getElementById('sg-bg-root');
              const scene = new THREE.Scene();
              const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
              camera.position.z = 60;

              const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
              renderer.setPixelRatio(window.devicePixelRatio || 1);
              renderer.setSize(window.innerWidth, window.innerHeight);
              renderer.domElement.id = 'sg-bg-canvas';
              container.appendChild(renderer.domElement);

              const count = {int(particle_count)};
              const geometry = new THREE.BufferGeometry();
              const positions = new Float32Array(count * 3);
              const speeds = new Float32Array(count);
              for (let i = 0; i < count; i++) {{
                positions[i*3 + 0] = (Math.random() - 0.5) * 180;
                positions[i*3 + 1] = (Math.random() - 0.5) * 100;
                positions[i*3 + 2] = (Math.random() - 0.5) * 160;
                speeds[i] = 0.05 + Math.random() * 0.25;
              }}
              geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

              const material = new THREE.PointsMaterial({{
                color: {particle_color_js},
                size: 1.6,
                transparent: true,
                opacity: 0.85,
                blending: THREE.AdditiveBlending,
                depthWrite: false
              }});
              const points = new THREE.Points(geometry, material);
              scene.add(points);

              scene.fog = new THREE.FogExp2(0x0a0f1a, 0.015);

              let t = 0;
              function animate(){{
                t += 0.005 * {float(speed)};
                camera.position.x = Math.sin(t * 0.7) * 6;
                camera.position.y = Math.cos(t * 0.5) * 3;
                camera.lookAt(0,0,0);

                const pos = geometry.attributes.position.array;
                for (let i = 0; i < count; i++) {{
                  const idx = i*3 + 1;
                  pos[idx] += Math.sin(t + i * 0.01) * speeds[i] * 0.3 * {float(speed)};
                  if (pos[idx] > 60) pos[idx] = -60;
                  if (pos[idx] < -60) pos[idx] = 60;
                }}
                geometry.attributes.position.needsUpdate = true;

                renderer.render(scene, camera);
                requestAnimationFrame(animate);
              }}
              animate();

              function onResize(){{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
              }}
              window.addEventListener('resize', onResize);
            }} catch (e) {{
              console.warn('3D background failed to initialize:', e);
            }}
          }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )

st.sidebar.header("Settings")
etherscan_key = st.sidebar.text_input("Etherscan API Key (optional)", type="password")
wallet = st.sidebar.text_input("Wallet address (0x...)")
use_sample = st.sidebar.checkbox("Use simulated sample features (no on-chain fetch)", value=False)

with st.sidebar.expander("Background 3D (beta)", expanded=True):
    enable_3d = st.checkbox("Enable", value=True)
    low_power = st.checkbox("Low-power mode (static)", value=False)
    particle_color = st.color_picker("Particle color", value="#6aa7ff")
    col_a, col_b = st.columns(2)
    with col_a:
        particle_count = st.slider("Particles", min_value=200, max_value=3000, value=1200, step=100)
    with col_b:
        speed = st.slider("Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    bg1 = st.color_picker("Background color 1", value="#0b1220")
    bg2 = st.color_picker("Background color 2", value="#060a13")
    bg3 = st.color_picker("Background color 3", value="#03060c")

if enable_3d:
    render_3d_background(enable=True, low_power=low_power, particle_count=particle_count, speed=speed,
                         particle_hex=particle_color, bg1=bg1, bg2=bg2, bg3=bg3)

# Model controls
st.sidebar.markdown("---")
st.sidebar.subheader("Model")
sg = ensure_model_ready()
if sg is not None:
    st.sidebar.success(f"Model ready: {MODEL_PATH.name}")
else:
    st.sidebar.warning("Model not available. Demo mode only.")

# Train from simulated features (if available)
st.sidebar.markdown("---")
st.sidebar.markdown("Need help? Make sure `sharkguard/core.py` and `utils/etherscan.py` are present.")

# Main area
st.write("Enter your Etherscan API key (optional) and a wallet address, then click Analyze. The built-in model loads automatically.")
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Analyze wallet")
    analyze = st.button("🔍 Analyze")

with col2:
    if SIM_FEATURES.exists():
        st.caption(f"Sim features: {SIM_FEATURES.name}")
    else:
        st.caption("No simulated features found")

# When Analyze is pressed
if analyze:
    if use_sample:
        # show a random row from simulated features CSV
        if SIM_FEATURES.exists():
            df_feats = pd.read_csv(SIM_FEATURES)
            st.success("Loaded simulated features — showing a random sample")
            st.dataframe(df_feats.sample(1).T)
        else:
            st.error("Simulated features file not present. Run: python data/simulate.py")
    else:
        if not wallet:
            st.error("Please enter a wallet address (0x...) or enable 'Use simulated sample features'.")
        else:
            # Fetch real blockchain data if API key provided
            txs = []
            current_balance = 0.0
            
            if etherscan_key:
                try:
                    with st.spinner("Fetching real blockchain data from Etherscan..."):
                        # Fetch recent transactions (last 10,000 or API limit)
                        txs = fetch_transactions(wallet, etherscan_key, sort="desc", offset=10000)
                        current_balance = fetch_account_balance(wallet, etherscan_key)
                        
                    if txs:
                        st.success(f"✅ Fetched {len(txs)} real transactions from blockchain")
                        st.info(f"💰 Current balance: {current_balance:.4f} ETH")
                    else:
                        st.warning("No transactions found for this address")
                        
                except Exception as e:
                    st.error("Etherscan fetch failed: " + str(e))
                    txs = []
            else:
                st.info("💡 **Tip**: Provide an Etherscan API key to analyze real blockchain data instead of demo mode.")
                st.info("🔗 Get a free API key at: https://etherscan.io/apis")

            # Convert to dataframe and extract features
            try:
                df = txs_to_dataframe(txs)
            except Exception as e:
                st.error("Failed to convert transactions to DataFrame: " + str(e))
                df = pd.DataFrame()

            # If no real data, create demo data based on wallet address
            if df.empty and not etherscan_key:
                feat = create_demo_features(wallet)
                st.info("🔍 Demo Mode: Using simulated features based on wallet address pattern")
            else:
                feat = extract_wallet_features(df, wallet)
            
            # TABBED OUTPUT UI
            tab_overview, tab_heur, tab_tx = st.tabs(["Overview", "Heuristics", "Transactions"])

            with tab_overview:
                st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
                st.subheader("Extracted features")
                st.json(feat)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
                st.subheader("Model Result")
                if sg is None:
                    st.warning("No model loaded. Demo mode only.")
                else:
                    try:
                        res = sg.predict_score(feat)
                        risk_pct = int(max(0.0, min(1.0, res['score'])) * 100)
                        label = res.get("label", "unknown").upper()
                        badge_color = "green" if label == "NORMAL" else "red"
                        st.markdown(f"<span class='sg-badge {badge_color}'> {label} </span>", unsafe_allow_html=True)
                        st.write("")
                        st.markdown("<div class='sg-riskbar'><div style='width:{}%'></div></div>".format(risk_pct), unsafe_allow_html=True)
                        st.caption(f"Suspicion score: {res['score']:.3f} (0 = normal, 1 = suspicious)")
                        with st.expander("Details"):
                            st.code(str({"raw": res.get("raw")}), language="text")
                    except Exception as e:
                        st.error("Model prediction failed: " + str(e))
                        st.text(traceback.format_exc())
                st.markdown("</div>", unsafe_allow_html=True)

            # Advanced Heuristic Analysis
            heuristic_analysis = analyze_wallet_heuristics(feat, df if not df.empty else None)

            with tab_heur:
                st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
                st.subheader("🔍 Advanced Heuristic Analysis")
                risk_color = {
                    'LOW': 'green',
                    'MEDIUM': 'orange', 
                    'HIGH': 'red'
                }.get(heuristic_analysis['risk_level'], 'gray')
                cols = st.columns(3)
                cols[0].markdown(f"<span class='sg-badge {risk_color}'>Risk: {heuristic_analysis['risk_level']}</span>", unsafe_allow_html=True)
                cols[1].metric("Risk Score", f"{heuristic_analysis['risk_score']:.2f}")
                cols[2].metric("Flags Detected", len(heuristic_analysis['flags']))

                if heuristic_analysis['explanations']:
                    st.write("**🚨 Risk Indicators:**")
                    for explanation in heuristic_analysis['explanations']:
                        st.write(f"• {explanation}")
                else:
                    st.success("✅ No significant risk indicators detected")

                if heuristic_analysis.get('behavioral_patterns'):
                    st.write("**📊 Behavioral Patterns:**")
                    patterns = ', '.join(heuristic_analysis['behavioral_patterns'])
                    st.info(f"Detected patterns: {patterns}")

                if heuristic_analysis.get('recommendations'):
                    st.write("**💡 Recommendations:**")
                    for rec in heuristic_analysis['recommendations']:
                        st.write(f"• {rec}")
                st.markdown("</div>", unsafe_allow_html=True)

            with tab_tx:
                st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
                st.subheader("📋 Transaction Analysis")
                if not df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Transactions", len(df))
                        if 'tx_success' in df.columns:
                            success_rate = df['tx_success'].mean()
                            st.metric("Success Rate", f"{success_rate:.1%}")
                    with col2:
                        if 'value_eth' in df.columns:
                            total_volume = df['value_eth'].sum()
                            st.metric("Total Volume", f"{total_volume:.4f} ETH")
                        if 'gasUsed' in df.columns and 'gasPrice' in df.columns:
                            total_gas_cost = (df['gasUsed'] * df['gasPrice'] / 1e18).sum()
                            st.metric("Total Gas Cost", f"{total_gas_cost:.4f} ETH")

                    st.write("**Recent Transactions (Last 20):**")
                    display_cols = ['timeStamp', 'from', 'to', 'value_eth']
                    if 'tx_success' in df.columns:
                        display_cols.append('tx_success')
                    if 'is_contract_call' in df.columns:
                        display_cols.append('is_contract_call')
                    display_df = df[display_cols].head(20).copy()
                    if 'timeStamp' in display_df.columns:
                        display_df['timeStamp'] = display_df['timeStamp'].dt.strftime('%Y-%m-%d %H:%M')
                    if 'value_eth' in display_df.columns:
                        display_df['value_eth'] = display_df['value_eth'].round(6)
                    if 'from' in display_df.columns:
                        display_df['from'] = display_df['from'].str[:10] + '...'
                    if 'to' in display_df.columns:
                        display_df['to'] = display_df['to'].str[:10] + '...'
                    st.dataframe(display_df, use_container_width=True)

                    if len(df) > 1 and 'timeStamp' in df.columns and 'value_eth' in df.columns:
                        import matplotlib.pyplot as plt
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        df_sorted = df.sort_values('timeStamp')
                        ax1.plot(df_sorted['timeStamp'], df_sorted['value_eth'], 'b-', alpha=0.7)
                        ax1.set_title('Transaction Values Over Time')
                        ax1.set_ylabel('Value (ETH)')
                        ax1.grid(True, alpha=0.3)
                        hourly_counts = df['timeStamp'].dt.hour.value_counts().sort_index()
                        ax2.bar(hourly_counts.index, hourly_counts.values, alpha=0.7, color='green')
                        ax2.set_title('Transaction Frequency by Hour')
                        ax2.set_xlabel('Hour of Day')
                        ax2.set_ylabel('Number of Transactions')
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("No transaction data available for analysis.")
                    if not etherscan_key:
                        st.info("💡 Provide an Etherscan API key to fetch real transaction data.")
                st.markdown("</div>", unsafe_allow_html=True)
