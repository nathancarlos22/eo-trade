import pandas as pd
import numpy as np
from binance.client import Client
import matplotlib.pyplot as plt
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import requests

load_dotenv()

chat_id = os.getenv('CHAT_ID')
token = os.getenv('TOKEN')

def sendMenssageTelegram(message):
    try:
        url_base = f'https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}'
        response = requests.get(url_base)
        print(message)
        result = response.json()['result']['message_id']
        return result
    except Exception as e:
        print(e)
        return None

# Função para calcular indicadores e sinais de compra/venda
def calculate_indicators(data, ema_short_period, ema_long_period):
    data['ema_short'] = data['Close'].ewm(span=ema_short_period, adjust=False).mean()
    data['ema_long'] = data['Close'].ewm(span=ema_long_period, adjust=False).mean()
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    data['signal'] = 0.0
    data['signal'] = np.where((data['ema_short'] > data['ema_long']) & (data['Close'] > data['vwap']), 1.0, data['signal'])
    data['signal'] = np.where((data['ema_short'] < data['ema_long']) & (data['Close'] < data['vwap']), -1.0, data['signal'])
    data['positions'] = data['signal'].diff()
    return data

# Função para executar o backtest
def backtest(data, initial_capital):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    portfolio = pd.DataFrame(index=data.index).fillna(0.0)
    positions['BTC'] = data['signal']
    portfolio['positions'] = (positions.multiply(data['Close'], axis=0))
    portfolio['cash'] = initial_capital - (positions.diff().multiply(data['Close'], axis=0)).cumsum()
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio

# Configurar a API da Binance
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')
client = Client(api_key, api_secret)

# Função para carregar dados históricos do BTC/USD da Binance
def get_historical_data(symbol, interval, period):
    klines = client.get_historical_klines(symbol, interval, period)
    data = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return data

# Carregar os dados históricos do BTC/USD com intervalo de 1 minuto
data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, '7 days ago UTC')

# Parâmetros iniciais
initial_capital = 10000.0
best_total = 0
best_ema_short_period = 5
best_ema_long_period = 20

# Testar diferentes combinações de períodos das EMAs
for ema_short_period in range(3, 15):
    for ema_long_period in range(15, 50):
        data_with_indicators = calculate_indicators(data.copy(), ema_short_period, ema_long_period)
        portfolio = backtest(data_with_indicators, initial_capital)
        final_total = portfolio['total'].iloc[-1]
        
        if final_total > best_total:
            best_total = final_total
            best_ema_short_period = ema_short_period
            best_ema_long_period = ema_long_period
            print(f"Novo melhor total: {best_total} com EMA Curta: {best_ema_short_period} e EMA Longa: {best_ema_long_period}")

# Calcular indicadores com os melhores parâmetros
data = calculate_indicators(data, best_ema_short_period, best_ema_long_period)

# Verificar se os dados foram carregados corretamente
if data.empty:
    print("Erro: Nenhum dado foi carregado.")
else:
    print(f"Dados carregados: {data.shape[0]} linhas")

# Executar backtest com os melhores parâmetros
portfolio = backtest(data, initial_capital)

# Inicializar a visualização em tempo real
plt.ion()  # Habilitar modo interativo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plotar as linhas iniciais
line_price, = ax1.plot(data.index[:1], data['Close'][:1], label='Preço BTC/USD', color='blue', lw=2)
line_ema_short, = ax1.plot(data.index[:1], data['ema_short'][:1], label=f'EMA Curta ({best_ema_short_period})', color='green', lw=2)
line_ema_long, = ax1.plot(data.index[:1], data['ema_long'][:1], label=f'EMA Longa ({best_ema_long_period})', color='red', lw=2)
line_vwap, = ax1.plot(data.index[:1], data['vwap'][:1], label='VWAP', color='purple', lw=2)
line_buy_signals, = ax1.plot([], [], '^', markersize=10, color='m', label='Sinal de Compra')
line_sell_signals, = ax1.plot([], [], 'v', markersize=10, color='k', label='Sinal de Venda')

line_capital, = ax2.plot(portfolio.index[:1], portfolio['total'][:1], label='Evolução do Capital', color='purple', lw=2)

ax1.legend()
ax1.grid()
ax2.legend()
ax2.grid()

fig.autofmt_xdate()

# Variáveis para armazenar o último sinal enviado
last_buy_signal_time = None
last_sell_signal_time = None

# Loop para obter dados novos a cada 1 minuto
while True:
    current_time = datetime.now()
    
    # Esperar até os segundos serem '00'
    while current_time.second != 0:
        # time.sleep(1)
        current_time = datetime.now()

    if data.empty:
        print("Erro: Nenhum dado foi carregado.")
    
    else:
        data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, '7 days ago UTC')
        print(f"Dados carregados: {data.shape[0]} linhas")

        # Calcular indicadores com os melhores parâmetros
        data = calculate_indicators(data, best_ema_short_period, best_ema_long_period)

        # Executar backtest com os melhores parâmetros
        portfolio = backtest(data, initial_capital)
        data = data.tail(100)
        portfolio = portfolio.tail(100)

        # Atualizar gráficos
        line_price.set_data(data.index, data['Close'])
        line_ema_short.set_data(data.index, data['ema_short'])
        line_ema_long.set_data(data.index, data['ema_long'])
        line_vwap.set_data(data.index, data['vwap'])

        buy_signals = data.loc[data['positions'] == 1.0]
        sell_signals = data.loc[data['positions'] == -1.0]

        line_buy_signals.set_data(buy_signals.index, buy_signals['Close'])
        line_sell_signals.set_data(sell_signals.index, sell_signals['Close'])

        line_capital.set_data(portfolio.index, portfolio['total'])

        # Verificar e enviar sinal de compra
        if not buy_signals.empty and buy_signals.index[-1] < data.index[-1] and buy_signals.index[-1] != last_buy_signal_time:
            last_buy_signal_time = buy_signals.index[-1]
            print(f"\nSinal de Compra em {last_buy_signal_time}: {buy_signals.tail(1)['Close'].values[0]}")
            sendMenssageTelegram(f"\nSinal de Compra em {last_buy_signal_time}: {buy_signals.tail(1)['Close'].values[0]}")

        # Verificar e enviar sinal de venda
        if not sell_signals.empty and sell_signals.index[-1] < data.index[-1] and sell_signals.index[-1] != last_sell_signal_time:
            last_sell_signal_time = sell_signals.index[-1]
            print(f"\nSinal de Venda em {last_sell_signal_time}: {sell_signals.tail(1)['Close'].values[0]}")
            sendMenssageTelegram(f"\nSinal de Venda em {last_sell_signal_time}: {sell_signals.tail(1)['Close'].values[0]}")

        # Adicionar anotações para sinais de compra e venda
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        for line in ax1.lines[4:]:
            line.remove()
        
        for index, row in buy_signals.iterrows():
            ax1.annotate(f'{row["Close"]:.2f}', xy=(index, row['Close']), xytext=(index, row['Close'] + 0.002),
                         arrowprops=dict(facecolor='green', shrink=0.05), fontsize=8, color='green')

        for index, row in sell_signals.iterrows():
            ax1.annotate(f'{row["Close"]:.2f}', xy=(index, row['Close']), xytext=(index, row['Close'] - 0.002),
                         arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8, color='red')

        # Ajustar limites dos eixos
        ax1.relim()
        ax1.autoscale_view()

        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw()
        plt.pause(0.01)

    # valor de fechamento do BTC/USD
    print(f"Valor de fechamento do BTC/USD: {data.tail(1)['Close'].values[0]}\n")
    print("Esperando 1 minutos para a próxima atualização...\n")
    

plt.ioff()  # Desabilitar modo interativo
plt.show()
