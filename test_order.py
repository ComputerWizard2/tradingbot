import asyncio
from metaapi_cloud_sdk import MetaApi

# --- CONFIG ---
TOKEN = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiIzNzQwNDQzNTA0MzIzMzU2MzRkYmQ5YWJmMjllZmU3NyIsImFjY2Vzc1J1bGVzIjpbeyJpZCI6InRyYWRpbmctYWNjb3VudC1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVzdC1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcnBjLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVhbC10aW1lLXN0cmVhbWluZy1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOndzOnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtZXRhc3RhdHMtYXBpIiwibWV0aG9kcyI6WyJtZXRhc3RhdHMtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6InJpc2stbWFuYWdlbWVudC1hcGkiLCJtZXRob2RzIjpbInJpc2stbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoiY29weWZhY3RvcnktYXBpIiwibWV0aG9kcyI6WyJjb3B5ZmFjdG9yeS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibXQtbWFuYWdlci1hcGkiLCJtZXRob2RzIjpbIm10LW1hbmFnZXItYXBpOnJlc3Q6ZGVhbGluZzoqOioiLCJtdC1tYW5hZ2VyLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJiaWxsaW5nLWFwaSIsIm1ldGhvZHMiOlsiYmlsbGluZy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfV0sImlnbm9yZVJhdGVMaW1pdHMiOmZhbHNlLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiMzc0MDQ0MzUwNDMyMzM1NjM0ZGJkOWFiZjI5ZWZlNzciLCJpYXQiOjE3NjYwMTM1NjUsImV4cCI6MTc3Mzc4OTU2NX0.YIb65lYBzMJMmc_F-BZtf3w70Ypjy0fLI7spcDM1w6oJrQ0sVGWtAmOg9xu5jAMI-LjL3cWG-yJQNIS4zE4dPOMX_avk48blkcJ6WzqbBMao5wZSMyP9c4-LqcmFBFN-BqAUcX6S08pJeH7SE1vFxGIglBzB1eY8k_e7NYqE05mpgWv2hASb5LDAU9bnI5BCPZvQNiKAimi2XLnQHXSQGf1IcBldhdAJp3VGsfrfZdcDqcXPep-tQraquLWEMd2MWSJ7cJ3_xzsQkpdCLz2U6g-p2MYi4x5bfsn6MhaifAr_QxkshHIF3aC_pF-UA_SUsu7PLZj5_i6rUrbQyKfhqc6rkXXfzCACJ5MO-7dA-xtVuBqWt5y6G1dAamQRuy3Kpl00BcoOFm65iVHW2BL_dzurix4SCAr_UFmCQxZlAc5TbEwkj5Vmg_DT2Lxck8nBYO-3A-F-seHStCq0n44cjUKehaM5vG9BvvsOYr960rGJORuF6_TvoVqslfrOqs1aR0rbt9Xwp-XawfMc-FDjafAMKtMAEo5TT-frUgT7ra9v0OWatewExbyu_NGb8JG2k5jdAVTd_wJwZGu9TQksMp3adq7Di81kazKqo3ZmpZRlbDOIbjlRYSaTmBuLyxz8kz6xbOmOplLeTmUW146Y0vJHnLtR-JBgVBU_1DZg4CQ"
ACCOUNT_ID = "4e8fb4f8-b282-4d0b-b836-225cc06f68d9"
SYMBOL = "XAUUSD"
VOLUME = 0.01
MAGIC_NUMBER = 234000

async def test_trade():
    api = MetaApi(TOKEN)
    try:
        account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
        print(f"🔄 Connecting to account...")
        
        await account.deploy()
        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()
        
        print(f"✅ Connected! Testing execution...")

        # 1. Place Buy Order
        print(f"🟢 Placing TEST BUY order ({VOLUME} lots)...")
        result = await connection.create_market_buy_order(SYMBOL, VOLUME, options={'magic': MAGIC_NUMBER})
        order_id = result['orderId']
        print(f"✅ Order Placed! Ticket: {order_id}")
        
        # 2. Wait 5 seconds
        print("⏳ Waiting 5 seconds...")
        await asyncio.sleep(5)
        
        # 3. Close Order
        # We need to find the position ID from the order, usually same or related.
        # But safest is to find the open position for this symbol.
        positions = await connection.get_positions()
        target_pos = None
        for pos in positions:
            if pos['id'] == order_id or (pos['symbol'] == SYMBOL and pos.get('magic') == MAGIC_NUMBER):
                target_pos = pos
                break
        
        if target_pos:
            print(f"🔻 Closing Position {target_pos['id']}...")
            close_res = await connection.close_position(target_pos['id'], options={})
            print(f"✅ Position Closed! Code: {close_res['stringCode']}")
        else:
            print("⚠️ Could not find position to close. It might have closed instantly or failed.")

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
    finally:
        print("🛑 Done.")

if __name__ == "__main__":
    asyncio.run(test_trade())
