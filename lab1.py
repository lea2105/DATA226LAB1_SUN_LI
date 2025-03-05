# Importing 
from airflow import DAG
from airflow.models import Variable
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

from datetime import timedelta
from datetime import datetime
import snowflake.connector
import requests

# Defining snowflake connection 
def return_snowflake_conn():

    # Initialize the SnowflakeHook
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    
    # Execute the query and fetch results
    conn = hook.get_conn()
    return conn.cursor()


#############
## EXTRACT ##
#############

@task 
def extract(apikey, num_of_days, stock_symbol):

	# Define API endpoint
	API_URL = "https://www.alphavantage.co/query"

	# Last 180 days date range
	today = datetime.today()  
	start_date = today - timedelta(days=num_of_days)

	# API request parameters
	params = {
		"function": "TIME_SERIES_DAILY",
		"symbol": stock_symbol,
		"start_date": start_date.strftime("%Y-%m-%d"), "end_date": today.strftime("%Y-%m-%d"), "interval": "daily",
		"apikey": apikey 
		}

	# Send API request
	response = requests.get(API_URL, params=params)


	# Extract data
	if response.status_code == 200: 
		data = response.json() # Convert response to JSON print("Data retrieved successfully!")
		return data 

	else:
		print(f"Error: {response.status_code}, Message: {response.text}")

	

###############
## TRANSFORM ##
###############

@task 
def transform(input_data, num_of_days, stock_symbol):
	time_series = input_data.get("Time Series (Daily)", {})

	# Initialize 
	stock_data = []
	today = datetime.today()
	start_date = today - timedelta(days=num_of_days)   # Get last n days

	# Populate 
	for date, values in time_series.items():
		if datetime.strptime(date, "%Y-%m-%d").date() >= start_date.date():
			stock_data.append({
				"symbol": stock_symbol,
				"date": date,
				"open": float(values["1. open"]), "close": float(values["4. close"]), "high": float(values["2. high"]), "low": float(values["3. low"]), "volume": int(values["5. volume"])
				})

	return stock_data




##########
## LOAD ##
##########

@task 
def load(cursor, target_table, stock_data_input):
	try: 
		cursor.execute("BEGIN;")
		cursor.execute(f"""
			CREATE OR REPLACE TABLE {target_table} ( 
			symbol STRING,
			date DATE,
			open FLOAT,
            close FLOAT,
            high FLOAT,
            low FLOAT,
            volume BIGINT,
            PRIMARY KEY (symbol, date)
            );
			""")
		cursor.execute(f"""DELETE FROM {target_table}""")

		for item in stock_data_input:
			# Get all data 
			symbol = item['symbol']
			date = item['date']
			open = item['open']
			close = item['close']
			high = item['high']
			low = item['low']
			volume = item['volume']

			# Insert 
			sql = f"INSERT INTO {target_table} (symbol, date, open, close, high, low, volume) VALUES ('{symbol}', '{date}', {open}, {close}, {high}, {low}, {volume})"
			cursor.execute(sql)

		cursor.execute("COMMIT;")

	except Exception as e:
		cursor.execute("ROLLBACK;") 
		print(e)
		raise(e)


###########
## TRAIN ##
###########


@task
def train(cur, train_input_table, train_view, forecast_function_name):
    """
     - Create a view with training related columns
     - Create a model with the view above
    """

    create_view_sql = f"""CREATE OR REPLACE VIEW {train_view} AS SELECT
        DATE, CLOSE, SYMBOL
        FROM {train_input_table};"""

    create_model_sql = f"""CREATE OR REPLACE SNOWFLAKE.ML.FORECAST {forecast_function_name} (
        INPUT_DATA => SYSTEM$REFERENCE('VIEW', '{train_view}'),
        SERIES_COLNAME => 'SYMBOL',
        TIMESTAMP_COLNAME => 'DATE',
        TARGET_COLNAME => 'CLOSE',
        CONFIG_OBJECT => {{ 'ON_ERROR': 'SKIP' }}
    );"""

    try:
        cur.execute(create_view_sql)
        cur.execute(create_model_sql)
        # Inspect the accuracy metrics of your model. 
        cur.execute(f"CALL {forecast_function_name}!SHOW_EVALUATION_METRICS();")
    except Exception as e:
        print(e)
        raise


#############
## PREDICT ##
#############

@task
def predict(cur, forecast_function_name, train_input_table, forecast_table, final_table):
    """
     - Generate predictions and store the results to a table named forecast_table.
     - Union your predictions with your historical data, then create the final table
    """
    make_prediction_sql = f"""BEGIN
        -- This is the step that creates your predictions.
        CALL {forecast_function_name}!FORECAST(
            FORECASTING_PERIODS => 7,
            -- Here we set your prediction interval.
            CONFIG_OBJECT => {{'prediction_interval': 0.95}}
        );
        -- These steps store your predictions to a table.
        LET x := SQLID;
        CREATE OR REPLACE TABLE {forecast_table} AS SELECT * FROM TABLE(RESULT_SCAN(:x));
    END;"""
    create_final_table_sql = f"""CREATE OR REPLACE TABLE {final_table} AS
	    -- Historical data (up to today)
	    SELECT SYMBOL, DATE, CLOSE AS actual, NULL AS forecast, NULL AS lower_bound, NULL AS upper_bound
	    FROM {train_input_table}

	    UNION ALL

	    -- Future forecasted data
	    SELECT 
	        TRIM(series) AS SYMBOL, 
	        ts::DATE AS DATE,  
	        NULL AS actual, 
	        forecast::FLOAT AS forecast,  
	        lower_bound::FLOAT AS lower_bound,  
	        upper_bound::FLOAT AS upper_bound  
	    FROM {forecast_table}
	    WHERE ts > (SELECT MAX(DATE) FROM {train_input_table});
		"""



    try:
        cur.execute(make_prediction_sql)
        cur.execute(create_final_table_sql)
    except Exception as e:
        print(e)
        raise




##########
## TASK ##
##########

# Connect to snowflake
cursor = return_snowflake_conn()

# Get API key from airflow 
api_key = Variable.get("api_key")

stock_symbol = "AAPL"
num_of_days = 180
target_table = "dev.raw.lab1_stock_data"
train_view = "dev.adhoc.lab1_train_view"
forecast_function_name = "dev.analytics.lab1_predict_function"
forecast_table = "dev.adhoc.lab1_forecast_table"
final_table = "dev.analytics.lab1_prediction_results"


with DAG(
    dag_id = 'Lab1Task',
    start_date = datetime(2025,2,28),
    catchup=False,
    tags=['predict', 'ELT'],
    schedule = '10 * * * *'
) as dag:

    data = extract(api_key, num_of_days, stock_symbol)
    transformed_data = transform(data, num_of_days, stock_symbol)
    load_task = load(cursor, target_table, transformed_data)
    train_task = train(cursor, target_table, train_view, forecast_function_name)
    predict_task = predict(cursor, forecast_function_name, target_table, forecast_table, final_table)

    # Dependency 
    data >> transformed_data >> load_task >> train_task >> predict_task



