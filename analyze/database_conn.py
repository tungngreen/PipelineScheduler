import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Hide SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'
conn_str = "postgresql://postgres:pipe@db_host:port/pipeline"


def get_table_names(conn, schema, keyword):
    query = text(f"""SELECT table_name FROM information_schema.tables
        WHERE table_schema = :schema AND table_name LIKE :keyword;""")
    table_names = pd.read_sql(query, conn, params={"schema": schema, "keyword": f"%{keyword}%"})
    return table_names['table_name'].tolist()


def align_timestamps(dataframes, bin_size=1e6):  # 1e6 = 1 second binning
    for df in dataframes:
        df['aligned_timestamp'] = (df['timestamps'] // bin_size) * bin_size
    merged_df = pd.concat(dataframes).groupby('aligned_timestamp').sum(numeric_only=True).reset_index()

    #set timestamps to passed minutes
    first_timestamp = merged_df['aligned_timestamp'].iloc[0]
    tmp = merged_df['aligned_timestamp']
    merged_df['aligned_timestamp'] = (tmp - first_timestamp) / (60 * 1e6)
    merged_df.loc[0, 'aligned_timestamp'] = 0
    return merged_df


def bucket_and_average(merged_df, columns, num_buckets=100, window_size=3):
    merged_df['bucket'] = pd.cut(merged_df['aligned_timestamp'], bins=num_buckets, labels=False)

    if len(columns) == 1:
        bucketed_df = merged_df.groupby('bucket').agg({
            'aligned_timestamp': 'min',
            columns[0]: 'mean'
        }).reset_index()
        bucketed_df[columns[0]] = bucketed_df[columns[0]].rolling(window=window_size, min_periods=1).mean()
        return bucketed_df

    # Group by the bucket and compute the mean in each bucket
    bucketed_df = merged_df.groupby('bucket').agg({
        'aligned_timestamp': 'min',
        columns[0]: 'mean',
        columns[1]: 'mean'
    }).reset_index()

    # Apply running average (rolling window) to smooth the values
    bucketed_df[columns[0]] = bucketed_df[columns[0]].rolling(window=window_size, min_periods=1).mean()
    bucketed_df[columns[1]] = bucketed_df[columns[1]].rolling(window=window_size, min_periods=1).mean()
    return bucketed_df


def arrival_metrics(schema):
    merged_df = arrival_query(schema)
    print(f"Total late requests: {merged_df['late_requests'].sum()}")
    print(f"Total queue drops: {merged_df['queue_drops'].sum()}")
    merged_df = bucket_and_average(merged_df, ['queue_drops', 'late_requests'])

    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['aligned_timestamp'], merged_df['queue_drops'], label='Queue Drops', color='orange')
    plt.plot(merged_df['aligned_timestamp'], merged_df['late_requests'], label='Late Requests', color='blue')
    plt.xlabel('Minutes Passed')
    plt.ylabel('# of Requests')
    plt.title('Missed Requests Over Time')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def avg_memory(exp, algos=['ppp', 'dis', 'jlf', 'rim'], edge_device_count=0):
    engine = create_engine(conn_str)
    memories = {}
    df = pd.DataFrame()
    for algo in algos:
        full_schema = f"{exp}_{algo}"
        with engine.connect() as connection:
            if edge_device_count == 0:
                table_names = get_table_names(connection, full_schema, 'serv_hw')
                for table in table_names:
                    query = text(
                        f"SELECT timestamps, mem_usage, gpu_0_mem_usage, gpu_1_mem_usage, gpu_2_mem_usage, gpu_3_mem_usage FROM {full_schema}.{table};")
                    df = pd.read_sql(query, connection)
                df = df.loc[
                    (df[['gpu_0_mem_usage', 'gpu_1_mem_usage', 'gpu_2_mem_usage', 'gpu_3_mem_usage']] != 0).any(axis=1)]
                df['total_gpu_mem'] = df[
                    ['gpu_0_mem_usage', 'gpu_1_mem_usage', 'gpu_2_mem_usage', 'gpu_3_mem_usage']].sum(axis=1)
            else:
                table_names = get_table_names(connection, full_schema, 'agx%_hw')
                table_names.extend(get_table_names(connection, full_schema, 'nx%_hw'))
                table_names.extend(get_table_names(connection, full_schema, 'orn%_hw'))
                df = pd.DataFrame()
                for table in table_names:
                    query = text(f"SELECT timestamps, mem_usage, gpu_0_mem_usage FROM {full_schema}.{table};")
                    df_tmp = pd.read_sql(query, connection)
                    df = pd.concat([df, df_tmp], axis=0)
                df = df.loc[(df[['gpu_0_mem_usage']] != 0).any(axis=1)]
                df['total_gpu_mem'] = df['gpu_0_mem_usage']
        if edge_device_count == 0:
            memories[algo] = [df['total_gpu_mem'].mean(), df['mem_usage'].mean()]
        else:
            memories[algo] = [df['total_gpu_mem'].mean() * edge_device_count, df['mem_usage'].mean() * edge_device_count]
    print(memories)
    return memories

def avg_edge_power(exp, algos=['fcpo', 'bce', 'ppp']):
    engine = create_engine(conn_str)
    powers = {}
    df = pd.DataFrame()
    for algo in algos:
        full_schema = f"{exp}_{algo}"
        with engine.connect() as connection:
            table_names = get_table_names(connection, full_schema, 'agx%_hw')
            table_names.extend(get_table_names(connection, full_schema, 'nx%_hw'))
            table_names.extend(get_table_names(connection, full_schema, 'orn%_hw'))
            df = pd.DataFrame()
            for table in table_names:
                query = text(f"SELECT timestamps, gpu_0_mem_usage, gpu_0_power_consumption FROM {full_schema}.{table};")
                df_tmp = pd.read_sql(query, connection)
                df = pd.concat([df, df_tmp], axis=0)
            df = df.loc[(df[['gpu_0_mem_usage']] != 0).any(axis=1)]
        powers[algo] = [df['gpu_0_power_consumption'].mean()]
    print(powers)
    return powers

def full_memory(schema, buckets):
    engine = create_engine(conn_str)
    with engine.connect() as connection:
        table_names = get_table_names(connection, schema, 'serv_hw')
        for table in table_names:
            query = text(f"SELECT timestamps, mem_usage, gpu_0_mem_usage, gpu_1_mem_usage, gpu_2_mem_usage, gpu_3_mem_usage FROM {schema}.{table};")
            df = pd.read_sql(query, connection)
    df = df.loc[(df[['gpu_0_mem_usage', 'gpu_1_mem_usage', 'gpu_2_mem_usage', 'gpu_3_mem_usage']] != 0).any(axis=1)]
    df['total_gpu_mem'] = df[['gpu_0_mem_usage', 'gpu_1_mem_usage', 'gpu_2_mem_usage', 'gpu_3_mem_usage']].sum(axis=1)
    df['aligned_timestamp'] = (df['timestamps'] // 1e6) * 1e6
    df = bucket_and_average(df, ['total_gpu_mem', 'mem_usage'], num_buckets=buckets)
    print([df['total_gpu_mem'].mean(), df['mem_usage'].mean()])
    return df


def arrival_query(schema):
    engine = create_engine(conn_str)
    dataframes = []
    with engine.connect() as connection:
        table_names = get_table_names(connection, schema, 'arr')
        for table in table_names:
            query = text(f"SELECT timestamps, late_requests, queue_drops FROM {schema}.{table};")
            df = pd.read_sql(query, connection)
            print(table + " late_requests: " + str(df['late_requests'].sum()) + " queue_drops: " + str(
                df['queue_drops'].sum()))
            dataframes.append(df)
    return align_timestamps(dataframes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema', type=str, default='')
    parser.add_argument('--mode', type=str, default='')
    args = parser.parse_args()

    if args.mode == 'arrival':
        arrival_metrics(args.schema)
    if args.mode == 'memory':
        avg_memory(args.schema)
    else:
        print("Invalid mode. Please choose valid mode.")
