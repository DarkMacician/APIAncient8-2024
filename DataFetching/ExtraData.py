import requests
from datetime import datetime, timedelta
import time

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://prompt:123@cluster0.admu7.mongodb.net/NeoFeed?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['NeoFeed']
gen = db['General']
ex = db['Extra']


# Function to get the current block number on BSC
def get_current_block_number_bsc():
    url = "https://api.bscscan.com/api"
    params = {
        "module": "proxy",
        "action": "eth_blockNumber",
        "apikey": "2X7CXUBXYZ4GYEHG6QATI8G6SHSBN7977G"  # Replace with your BscScan API key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return int(data['result'], 16)  # Convert from hex to decimal

# Function to get the gas fee on BSC
def get_gas_fee_bsc():
    url = "https://api.bscscan.com/api"
    params = {
        "module": "gastracker",
        "action": "gasoracle",
        "apikey": "2X7CXUBXYZ4GYEHG6QATI8G6SHSBN7977G"  # Replace with your BscScan API key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data['result']['ProposeGasPrice']  # ProposeGasPrice represents the average gas price

# Function to get the transaction count for a specific block on BSC
def get_transaction_count_bsc(block_number):
    url = "https://api.bscscan.com/api"
    params = {
        "module": "proxy",
        "action": "eth_getBlockTransactionCountByNumber",
        "tag": hex(block_number),  # Convert the block number to hex format
        "apikey": "2X7CXUBXYZ4GYEHG6QATI8G6SHSBN7977G"  # Replace with your BscScan API key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return int(data['result'], 16)  # Convert the result from hex to decimal

# Fetch the gas fee and transaction count every minute for the next hour
def main():
    start_date = datetime.now()  # Start from the current time
    end_date = start_date + timedelta(hours=1)  # End after 1 hour

    # Get the current block number
    current_block_number = get_current_block_number_bsc()

    # Start the loop to fetch data every minute for the next hour
    current_time = start_date
    while current_time <= end_date:
        try:
            gas_fee = get_gas_fee_bsc()
            transaction_count = get_transaction_count_bsc(current_block_number)
            timestamp = current_time.strftime('%Y-%m-%d %H:%M')

            # Save data to MongoDB
            ex.insert_one({
                "block_number": current_block_number,
                "gas_fee_gwei": gas_fee,
                "transaction_count": transaction_count,
                "timestamp": timestamp
            })

            print(
                f"Data recorded for {timestamp} - Block: {current_block_number}, Gas Fee: {gas_fee} Gwei, Transactions: {transaction_count}")

            # Increment by 1 minute
            current_time += timedelta(minutes=1)

            # Update the current block number every time (or every few blocks depending on your requirements)
            current_block_number = get_current_block_number_bsc()

        except Exception as e:
            print(f"An error occurred: {e}")
            break

        # Wait for 1 minute
        time.sleep(60)

if __name__ == "__main__":
    main()
