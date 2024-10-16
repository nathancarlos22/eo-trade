import pandas as pd
import numpy as np
from binance.client import Client
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import requests
import schedule
import threading

load_dotenv()

# Carregar as variáveis de ambiente
chat_id = os.getenv('CHAT_ID')
token = os.getenv('TOKEN')
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Inicializar o cliente da Binance
client = Client(api_key, api_secret)

def send_telegram_message(message):
    """Envia uma mensagem para o Telegram."""
    try:
        print(f"Enviando mensagem para o Telegram: {message}")
        url_base = f'https://api.telegram.org/bot{token}/sendMessage'
        params = {'chat_id': chat_id, 'text': message}
        response = requests.get(url_base, params=params)
        response.raise_for_status()
        print("Mensagem enviada com sucesso.")
        return response.json()['result']['message_id']
    except requests.RequestException as e:
        print(f"Erro ao enviar mensagem para o Telegram: {e}")
        return None

# Função para buscar o histórico dos últimos 30 dias (para calcular corretamente o RSI)
def get_crypto_history(symbol):
    print(f"Buscando histórico de {symbol}...")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)
    
    # Converte as datas para o formato necessário (milissegundos)
    start_str = int(start_time.timestamp() * 1000)
    end_str = int(end_time.timestamp() * 1000)
    
    # Obtém o histórico de preços de 30 dias (intervalo de 1 hora para mais dados)
    try:
        # A Binance limita o número de candles retornados por requisição, por isso precisamos iterar
        klines = []
        limit = 1000  # Máximo permitido pela API
        while start_str < end_str:
            temp_klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, startTime=start_str, endTime=end_str, limit=limit)
            if not temp_klines:
                break
            klines.extend(temp_klines)
            start_str = temp_klines[-1][0] + 1  # Evitar duplicatas
            time.sleep(0.1)  # Pequena pausa para respeitar limites da API
        print(f"Recebido {len(klines)} candles para {symbol}")
    except Exception as e:
        print(f"Erro ao obter dados para {symbol}: {e}")
        return pd.DataFrame()
        
    data = []
    for candle in klines:
        open_time = datetime.utcfromtimestamp(candle[0] / 1000)
        close_price = float(candle[4])
        data.append([symbol, open_time, close_price])
    
    return pd.DataFrame(data, columns=['Ativo', 'Data', 'Preço de Fechamento'])

# Função para calcular o RSI (usando um período de 14 dias)
def calculate_rsi(prices, period=14):
    print(f"Calculando RSI com período de {period}...")
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=period).mean()
    loss = down.rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    print(f"RSI calculado.")
    return rsi

# Variável global para manter o último RSI notificado
last_notified_data = {}

# Função para verificar o RSI e notificar via Telegram se for menor que 20 e diferente do último valor notificado
def check_rsi():
    global last_notified_data
    print("Iniciando verificação do RSI...")

    # Obtém todas as criptomoedas disponíveis
    try:
        exchange_info = client.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']  # Apenas para pares com USDT que estão ativos
        print(f"Verificando RSI para {len(symbols)} símbolos.")
    except Exception as e:
        print(f"Erro ao obter informações de troca: {e}")
        return

    # Processar os símbolos em threads para melhorar a performance
    threads = []
    for symbol in symbols:
        t = threading.Thread(target=process_symbol, args=(symbol,))
        threads.append(t)
        t.start()
        time.sleep(0.05)  # Pausa para evitar sobrecarga de threads

    for t in threads:
        t.join()

def process_symbol(symbol):
    global last_notified_data
    try:
        df_crypto = get_crypto_history(symbol)
        if df_crypto.empty:
            print(f"Nenhum dado retornado para {symbol}")
            return
        
        # Calcula o RSI
        df_crypto['RSI'] = calculate_rsi(df_crypto['Preço de Fechamento'])
        
        # Verifica o último valor de RSI
        last_rsi = df_crypto['RSI'].iloc[-1]
        last_time = df_crypto['Data'].iloc[-1]  # Último horário da amostra de dados
        
        # Verifica se o RSI é menor que 20, diferente do último RSI notificado e se o horário mudou
        if last_rsi < 20:
            print(f"RSI de {symbol} está abaixo de 20 (RSI = {last_rsi:.2f}) no horário {last_time}")
            notified = last_notified_data.get(symbol, {})
            if notified.get('rsi') != last_rsi or notified.get('time') != last_time:
                # Atualiza o valor do último RSI notificado e o horário para o ativo
                last_notified_data[symbol] = {'rsi': last_rsi, 'time': last_time}
                
                # Envia notificação
                message = f"⚠️ RSI do ativo {symbol} em {last_time} está abaixo de 20! (RSI = {last_rsi:.2f})"
                send_telegram_message(message)
        else:
            print(f"RSI de {symbol} está acima de 20 (RSI = {last_rsi:.2f}) no horário {last_time}")
    except Exception as e:
        print(f"Erro ao processar o ativo {symbol}: {e}")

# Agendamento para rodar a cada minuto
schedule.every(1).minutes.do(check_rsi)

# Loop para manter o script rodando e verificando
while True:
    schedule.run_pending()
    time.sleep(1)
