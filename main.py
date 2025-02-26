import yaml
import psutil
import os
import time
from scapy.all import sniff, wrpcap, rdpcap
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras import regularizers
from datetime import datetime
import numpy as np


def get_interface_by_type(interface_type):
    """
    Возвращает имя интерфейса по указанному типу (wi-fi или ethernet).
    :param interface_type: "wi-fi" или "ethernet"
    :return: Имя интерфейса или None, если подходящий интерфейс не найден.
    """
    # Сопоставление типов интерфейсов с ключевыми словами в их именах
    interface_keywords = {
        "wi-fi": ["wireless", "wi-fi", "беспроводная сеть"],
        "ethernet": ["ethernet", "lan", "подключение по локальной сети"]
    }

    # Получаем список всех интерфейсов
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    # Ищем подходящий интерфейс
    for iface, addresses in addrs.items():
        # Проверяем, активен ли интерфейс
        if iface in stats and stats[iface].isup:
            # Проверяем, соответствует ли интерфейс указанному типу
            for keyword in interface_keywords.get(interface_type.lower(), []):
                if keyword.lower() in iface.lower():
                    return iface
    return None


# Загрузка конфигурационного файла
def load_config(config_file="config.yaml"):
    """
    Загружает конфигурацию из JSON-файла.
    """
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

# 1. Сбор сетевого трафика
def continuous_capture(output_file="traffic.pcap", interval=300, total_duration=86400, iface=None):
    """
    Непрерывный сбор трафика с сохранением данных через заданные интервалы.
    """
    print(f"Starting continuous traffic capture for {total_duration} seconds on interface {iface}...")
    start_time = time.time()
    packets = []

    while time.time() - start_time < total_duration:
        new_packets = sniff(timeout=interval, iface=iface)
        packets.extend(new_packets)
        wrpcap(output_file, packets)
        print(f"Saved {len(new_packets)} new packets to {output_file}. Total packets: {len(packets)}")

    print(f"Traffic capture completed. Total packets captured: {len(packets)}")

# 2. Оцифровка данных
def packet_to_features(packet):
    features = {
        'src_ip': packet['IP'].src if 'IP' in packet else '0.0.0.0',
        'dst_ip': packet['IP'].dst if 'IP' in packet else '0.0.0.0',
        'src_port': packet['TCP'].sport if 'TCP' in packet else 0,
        'dst_port': packet['TCP'].dport if 'TCP' in packet else 0,
        'packet_length': len(packet)
    }
    return features

def pcap_to_dataframe(pcap_file):
    packets = rdpcap(pcap_file)
    data = [packet_to_features(packet) for packet in packets]
    return pd.DataFrame(data)

# 3. Построение автоэнкодера
def build_autoencoder(input_dim, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(input_layer)
    encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
    encoded = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(encoded)
    encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(encoded)
    decoded = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(encoded)
    decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
    decoded = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(decoded)
    decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mae')
    return autoencoder

# 4. Обучение автоэнкодера
def train_autoencoder(df, model_path="autoencoder_model.h5", scaler_path="scaler.save"):
    label_encoder = LabelEncoder()
    df['src_ip'] = label_encoder.fit_transform(df['src_ip'])
    df['dst_ip'] = label_encoder.fit_transform(df['dst_ip'])
    X = df.drop(columns=['label']) if 'label' in df.columns else df
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)
    autoencoder.save(model_path)
    print(f"Autoencoder model saved to {model_path}")
    return autoencoder, scaler

# 5. Вычисление ошибок восстановления
def calculate_reconstruction_errors(model, data):
    reconstructed_data = model.predict(data)
    errors = np.mean(np.square(data - reconstructed_data), axis=1)
    return errors

# 6. Выбор порога аномальности
def select_threshold(errors, percentile=95):
    threshold = np.percentile(errors, percentile)
    return threshold

# 7. Обнаружение аномалий
def detect_anomalies(pcap_file, model_path="autoencoder_model.h5", scaler_path="scaler.save", threshold=None):
    custom_objects = {'mae': tf.keras.losses.MeanSquaredError()}
    autoencoder = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    df = pcap_to_dataframe(pcap_file)
    label_encoder = LabelEncoder()
    df['src_ip'] = label_encoder.fit_transform(df['src_ip'])
    df['dst_ip'] = label_encoder.fit_transform(df['dst_ip'])
    X = scaler.transform(df)
    errors = calculate_reconstruction_errors(autoencoder, X)
    if threshold is None:
        threshold = select_threshold(errors, percentile=95)
        print(f"Selected threshold: {threshold}")
    anomalies = np.where(errors > threshold)[0]
    print(f"Detected {len(anomalies)} anomalies")
    return anomalies, errors, threshold

# 8. Формирование отчетов
def generate_report(anomalies, errors, threshold, output_file):
    """
    Сохраняет отчет об аномалиях в файл.
    """
    with open(output_file, 'w') as f:
        f.write(f"Detected {len(anomalies)} anomalies\n")
        f.write(f"Threshold: {threshold}\n")
        for idx in anomalies:
            f.write(f"Anomaly detected at packet index {idx} with error {errors[idx]:.4f}\n")
    print(f"Report saved to {output_file}")

# 9. Визуализация ошибок
def plot_errors(errors, threshold):
    plt.hist(errors, bins=50, density=True)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.show()

# 10. Непрерывный сбор и проверка данных
def continuous_anomaly_detection(model_path="autoencoder_model.h5", scaler_path="scaler.save", capture_duration=60, check_interval=300, iface=None):
    custom_objects = {'mae': tf.keras.losses.MeanSquaredError()}
    autoencoder = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Перекомпилируем модель для устранения предупреждения о метриках
    autoencoder.compile(optimizer='adam', loss='mae')

    scaler = joblib.load(scaler_path)

    # Создаем папки для отчетов и тестовых данных, если они не существуют
    anomaly_report_dir = "report/anomaly_report"
    test_data_dir = "report/test_data"
    if not os.path.exists(anomaly_report_dir):
        os.makedirs(anomaly_report_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    while True:
        # Генерируем уникальную временную метку
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Имя файла для тестовых данных
        test_file = os.path.join(test_data_dir, f"test_traffic_{timestamp}.pcap")

        # Сбор данных
        continuous_capture(output_file=test_file, interval=capture_duration, total_duration=capture_duration, iface=iface)

        # Оцифровка данных
        df = pcap_to_dataframe(test_file)
        label_encoder = LabelEncoder()
        df['src_ip'] = label_encoder.fit_transform(df['src_ip'])
        df['dst_ip'] = label_encoder.fit_transform(df['dst_ip'])
        X = scaler.transform(df)

        # Проверка на аномалии
        errors = calculate_reconstruction_errors(autoencoder, X)
        threshold = select_threshold(errors, percentile=95)
        anomalies = np.where(errors > threshold)[0]

        # Имя файла для отчета
        report_file = os.path.join(anomaly_report_dir, f"anomaly_report_{timestamp}.txt")

        # Формирование отчета
        generate_report(anomalies, errors, threshold, output_file=report_file)

        # Вывод в консоль
        print(f"Checked {len(df)} packets. Detected {len(anomalies)} anomalies.")

        # Ожидание перед следующей проверкой
        time.sleep(check_interval)

# Основной скрипт
if __name__ == "__main__":

    config = load_config()
    # Пути к файлам модели и scaler
    model_path = config['path']['model_path']
    scaler_path = config['path']['scaler_path']

    # network_settings
    interface_type = config['network_settings']['interface_type']
    interval = config['network_settings']['interval']
    total_duration = config['network_settings']['total_duration']
    capture_duration = config['network_settings']['capture_duration']
    check_interval =  config['network_settings']['check_interval']

    iface = get_interface_by_type(interface_type)

    if iface:
        print(f"Selected interface for {interface_type}: {iface}")
    else:
        print(f"No active {interface_type} interface found.")
        exit(1)

    # Проверка наличия файлов модели и scaler
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading pre-trained model and scaler...")
        # Загрузка модели и scaler
        custom_objects = {'mae': tf.keras.losses.MeanSquaredError()}
        autoencoder = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        scaler = joblib.load(scaler_path)
    else:
        print("Pre-trained model not found. Starting training process...")
        # Сбор обучающей выборки (24 часа по умолчанию)

        print(f"Using interface for training: {iface}")
        continuous_capture(output_file="training_traffic.pcap", interval=interval, total_duration=total_duration, iface=iface)

        # Оцифровка данных
        df = pcap_to_dataframe("training_traffic.pcap")

        # Обучение автоэнкодера
        autoencoder, scaler = train_autoencoder(df, model_path=model_path, scaler_path=scaler_path)

    # Непрерывный сбор и проверка данных
    print(f"Using interface for testing: {iface}")
    continuous_anomaly_detection(model_path=model_path, scaler_path=scaler_path, capture_duration=capture_duration, check_interval=check_interval, iface=iface)
