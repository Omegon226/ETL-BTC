from init_influxdb import init_influxdb
from init_qdrant import init_qdrant


if __name__ == "__main__":
    print("Инциализация InfluxDB")
    init_influxdb()
    print("Инциализация Qdrant")
    init_qdrant()
