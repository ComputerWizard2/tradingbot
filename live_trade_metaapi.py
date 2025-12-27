import asyncio
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from stable_baselines3 import PPO
from metaapi_cloud_sdk import MetaApi

from features.make_features import compute_features

# Suppress MetaAPI internal error logs completely
import warnings
warnings.filterwarnings('ignore')

# Set all MetaAPI loggers to CRITICAL to hide subscription retry errors
for logger_name in ['metaapi_cloud_sdk', 'metaapi_cloud_sdk.clients',
                    'metaapi_cloud_sdk.clients.metaapi',
                    'metaapi_cloud_sdk.clients.metaapi.subscription_manager']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).disabled = True

# --- USER CONFIG (FILL THESE) ---
TOKEN = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiIzNzQwNDQzNTA0MzIzMzU2MzRkYmQ5YWJmMjllZmU3NyIsImFjY2Vzc1J1bGVzIjpbeyJpZCI6InRyYWRpbmctYWNjb3VudC1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVzdC1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcnBjLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVhbC10aW1lLXN0cmVhbWluZy1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOndzOnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtZXRhc3RhdHMtYXBpIiwibWV0aG9kcyI6WyJtZXRhc3RhdHMtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6InJpc2stbWFuYWdlbWVudC1hcGkiLCJtZXRob2RzIjpbInJpc2stbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoiY29weWZhY3RvcnktYXBpIiwibWV0aG9kcyI6WyJjb3B5ZmFjdG9yeS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibXQtbWFuYWdlci1hcGkiLCJtZXRob2RzIjpbIm10LW1hbmFnZXItYXBpOnJlc3Q6ZGVhbGluZzoqOioiLCJtdC1tYW5hZ2VyLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJiaWxsaW5nLWFwaSIsIm1ldGhvZHMiOlsiYmlsbGluZy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfV0sImlnbm9yZVJhdGVMaW1pdHMiOmZhbHNlLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiMzc0MDQ0MzUwNDMyMzM1NjM0ZGJkOWFiZjI5ZWZlNzciLCJpYXQiOjE3NjYwMTM1NjUsImV4cCI6MTc3Mzc4OTU2NX0.YIb65lYBzMJMmc_F-BZtf3w70Ypjy0fLI7spcDM1w6oJrQ0sVGWtAmOg9xu5jAMI-LjL3cWG-yJQNIS4zE4dPOMX_avk48blkcJ6WzqbBMao5wZSMyP9c4-LqcmFBFN-BqAUcX6S08pJeH7SE1vFxGIglBzB1eY8k_e7NYqE05mpgWv2hASb5LDAU9bnI5BCPZvQNiKAimi2XLnQHXSQGf1IcBldhdAJp3VGsfrfZdcDqcXPep-tQraquLWEMd2MWSJ7cJ3_xzsQkpdCLz2U6g-p2MYi4x5bfsn6MhaifAr_QxkshHIF3aC_pF-UA_SUsu7PLZj5_i6rUrbQyKfhqc6rkXXfzCACJ5MO-7dA-xtVuBqWt5y6G1dAamQRuy3Kpl00BcoOFm65iVHW2BL_dzurix4SCAr_UFmCQxZlAc5TbEwkj5Vmg_DT2Lxck8nBYO-3A-F-seHStCq0n44cjUKehaM5vG9BvvsOYr960rGJORuF6_TvoVqslfrOqs1aR0rbt9Xwp-XawfMc-FDjafAMKtMAEo5TT-frUgT7ra9v0OWatewExbyu_NGb8JG2k5jdAVTd_wJwZGu9TQksMp3adq7Di81kazKqo3ZmpZRlbDOIbjlRYSaTmBuLyxz8kz6xbOmOplLeTmUW146Y0vJHnLtR-JBgVBU_1DZg4CQ"
ACCOUNT_ID = "4e8fb4f8-b282-4d0b-b836-225cc06f68d9"

# --- STRATEGY CONFIG ---
SYMBOL = "XAUUSD"
TIMEFRAME = "1h"
VOLUME = 0.01
MODEL_PATH = "train/ppo_xauusd_latest.zip"
WINDOW = 64
MAGIC_NUMBER = 234000

async def get_market_data(account, symbol, n=1000, max_retries=3):
    """Fetch recent candles from MetaAPI Account with retry logic"""
    for attempt in range(max_retries):
        try:
            from datetime import datetime, timedelta
            # Fetch enough history for stable feature normalization
            start_time = datetime.now() - timedelta(days=60)
            candles = await asyncio.wait_for(
                account.get_historical_candles(symbol, TIMEFRAME, start_time, limit=n),
                timeout=30.0
            )
            if not candles:
                if attempt < max_retries - 1:
                    print(f"⚠️ No candles received, retrying... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2)
                    continue
                return None
            data = []
            for c in candles:
                data.append({
                    'time': c['time'],
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c['tickVolume']
                })
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(f"⚠️ Data fetch timeout, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
            else:
                print(f"❌ Failed to fetch data after {max_retries} attempts")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Error fetching data: {e}, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
            else:
                print(f"❌ Error fetching data after {max_retries} attempts: {e}")
                return None
    return None

async def get_current_position(connection):
    try:
        positions = await connection.get_positions()
        for pos in positions:
            if pos['symbol'] == SYMBOL and pos.get('magic') == MAGIC_NUMBER:
                if pos['type'] == 'POSITION_TYPE_BUY':
                    return 1, pos['id']
        return 0, None
    except Exception as e:
        print(f"❌ Error checking positions: {e}")
        return 0, None

async def run_step(account, connection, model):
    df = await get_market_data(account, SYMBOL, n=1000)
    if df is None: return

    _, feats, _ = compute_features(df)
    if len(feats) < WINDOW:
        print("⏳ Not enough data yet...")
        return
    
    obs_features = feats[-WINDOW:]
    current_pos_type, pos_id = await get_current_position(connection)
    obs = np.concatenate([obs_features.reshape(-1), np.array([current_pos_type], dtype=np.float32)])
    
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Pos: {current_pos_type} | Action: {action} ({'Long' if action==1 else 'Flat'})")
    
    if action == current_pos_type:
        pass
    elif action == 1 and current_pos_type == 0:
        print("🟢 Opening Long Position...")
        try:
            result = await connection.create_market_buy_order(SYMBOL, VOLUME, options={'magic': MAGIC_NUMBER})
            print(f"✅ Order Sent: {result['orderId']}")
        except Exception as e:
            print(f"❌ Order Failed: {e}")
    elif action == 0 and current_pos_type == 1:
        print("🔻 Closing Long Position...")
        try:
            result = await connection.close_position(pos_id, options={})
            print(f"✅ Closed: {result['orderId']}")
        except Exception as e:
            print(f"❌ Close Failed: {e}")

async def trade_loop():
    if TOKEN == "YOUR_METAAPI_TOKEN_HERE":
        print("❌ Please edit the script and set your TOKEN and ACCOUNT_ID!")
        return

    # Initialize MetaApi - let it auto-detect region from account
    api = MetaApi(TOKEN)
    try:
        account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

        # Get account region info
        account_region = getattr(account, 'region', 'unknown')
        print(f"🔄 Connecting to account {ACCOUNT_ID} (Region: {account_region})...")

        # Ensure account is deployed
        initial_state = account.state
        print(f"📊 Account state: {initial_state}")

        if initial_state != 'DEPLOYED':
            print(f"🚀 Deploying account (current state: {initial_state})...")
            await account.deploy()

            # Wait for deployment to complete
            for i in range(30):  # Wait up to 60 seconds
                await asyncio.sleep(2)
                await account.reload()
                current_state = account.state
                print(f"⏳ Deployment status: {current_state}")
                if current_state == 'DEPLOYED':
                    break
            else:
                raise Exception("Account deployment timed out after 60 seconds")

        print(f"✅ Account is DEPLOYED")

        # Wait additional time for broker connection to stabilize
        print("⏳ Waiting for broker connection to stabilize...")
        await asyncio.sleep(10)

        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()

        # Add delay to ensure subscription is fully established
        print("⏳ Waiting for subscription to stabilize...")
        await asyncio.sleep(10)

        # Test connection by fetching account information with retries
        for attempt in range(5):
            try:
                account_info = await asyncio.wait_for(
                    connection.get_account_information(),
                    timeout=30.0
                )
                print(f"✅ Connected to {account.name}! Balance: ${account_info.get('balance', 'N/A')}")
                break
            except Exception as e:
                if attempt < 4:
                    print(f"⚠️ Connection test failed (attempt {attempt + 1}/5): {e}")
                    await asyncio.sleep(5)
                else:
                    raise Exception(f"Connection test failed after 5 attempts: {e}")

        model = PPO.load(MODEL_PATH)
        print(f"🧠 Model loaded: {MODEL_PATH}")
        print("🚀 Starting Live Trading Loop (Ctrl+C to stop)...")

        while True:
            try:
                await asyncio.wait_for(run_step(account, connection, model), timeout=60.0)
            except asyncio.TimeoutError:
                print("⚠️ Network timed out. Retrying...")
            except Exception as e:
                error_msg = str(e)
                if "not connected" in error_msg.lower() or "timeout" in error_msg.lower():
                    print("⚠️ Connection issue detected, attempting to reconnect...")
                    try:
                        await connection.connect()
                        await connection.wait_synchronized()
                        await asyncio.sleep(3)
                        print("✅ Reconnected!")
                    except Exception as reconnect_error:
                        print(f"❌ Reconnection failed: {reconnect_error}")
                else:
                    print(f"⚠️ Error in loop: {e}")
            await asyncio.sleep(10)

    except Exception as e:
        print(f"💥 Critical Error: {e}")
    finally:
        print("🛑 Disconnecting...")

if __name__ == "__main__":
    try:
        asyncio.run(trade_loop())
    except KeyboardInterrupt:
        print("Stopped by user.")