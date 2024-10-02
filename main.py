import pandas as pd
import numpy as np
from binance.client import Client
import matplotlib.pyplot as plt
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
from matplotlib.animation import FuncAnimation

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

def calculate_indicators(data):
    """Calcula as EMAs necessárias para a estratégia."""
    # Calcular EMAs de 5 e 21 períodos
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    return data

def implement_strategy(data, stop_loss_diff, take_profit_diff, initial_capital):
    """Implementa a estratégia com stop-loss e take-profit."""
    data = data.copy()
    data['Position'] = 0
    data['Trade'] = 0
    data['Capital'] = initial_capital
    data['Stop_Loss'] = np.nan
    data['Take_Profit'] = np.nan

    position_active = False
    position_type = None  # 'long' ou 'short'
    entry_price = 0
    btc_amount = 0
    capital = initial_capital

    for i in range(1, len(data)):
        current_price = data['Close'].iloc[i]
        prev_EMA_5 = data['EMA_5'].iloc[i - 1]
        prev_EMA_21 = data['EMA_21'].iloc[i - 1]
        current_EMA_5 = data['EMA_5'].iloc[i]
        current_EMA_21 = data['EMA_21'].iloc[i]

        # Verificar se a posição está ativa
        # if position_active:
        #     if position_type == 'long':
        #         # Verificar stop loss para posição longa
        #         if current_price <= entry_price - stop_loss_diff:
        #             data.at[data.index[i], 'Stop_Loss'] = current_price
        #             capital = btc_amount * current_price
        #             btc_amount = 0
        #             position_active = False
        #             position_type = None
        #             data.at[data.index[i], 'Position'] = 0
        #         # Verificar take profit para posição longa
        #         elif current_price >= entry_price + take_profit_diff:
        #             data.at[data.index[i], 'Take_Profit'] = current_price
        #             capital = btc_amount * current_price
        #             btc_amount = 0
        #             position_active = False
        #             position_type = None
        #             data.at[data.index[i], 'Position'] = 0
        #     elif position_type == 'short':
        #         # Verificar stop loss para posição vendida
        #         if current_price >= entry_price + stop_loss_diff:
        #             data.at[data.index[i], 'Stop_Loss'] = current_price
        #             profit = btc_amount * current_price
        #             capital += profit
        #             btc_amount = 0
        #             position_active = False
        #             position_type = None
        #             data.at[data.index[i], 'Position'] = 0
        #         # Verificar take profit para posição vendida
        #         elif current_price <= entry_price - take_profit_diff:
        #             data.at[data.index[i], 'Take_Profit'] = current_price
        #             profit = btc_amount * current_price
        #             capital += profit
        #             btc_amount = 0
        #             position_active = False
        #             position_type = None
        #             data.at[data.index[i], 'Position'] = 0

        # else:
            # Verificar cruzamento de EMAs para entrar na posição longa
        if prev_EMA_5 <= prev_EMA_21 and current_EMA_5 > current_EMA_21:
            entry_price = current_price
            btc_amount = capital / entry_price
            capital = 0
            position_active = True
            position_type = 'long'
            data.at[data.index[i], 'Position'] = 1
            data.at[data.index[i], 'Trade'] = 1  # Compra

        # Verificar cruzamento de EMAs para entrar na posição vendida (short)
        elif prev_EMA_5 >= prev_EMA_21 and current_EMA_5 < current_EMA_21:
            entry_price = current_price
            btc_amount = - (capital / entry_price)  # btc_amount negativo para posição vendida
            capital += abs(btc_amount) * entry_price  # Recebe capital da venda a descoberto
            position_active = True
            position_type = 'short'
            data.at[data.index[i], 'Position'] = -1
            data.at[data.index[i], 'Trade'] = -1  # Venda

    # Atualizar o capital ao longo do tempo
    if position_active:
        if position_type == 'long':
            total_capital = btc_amount * current_price
        elif position_type == 'short':
            total_capital = capital + btc_amount * current_price
        data.at[data.index[i], 'Capital'] = total_capital
    else:
        data.at[data.index[i], 'Capital'] = capital

    return data

def get_historical_data(symbol, interval, period):
    """Carrega dados históricos do Binance."""
    print(f"Carregando dados históricos para {symbol} com intervalo {interval} e período {period}.")
    klines = client.get_historical_klines(symbol, interval, period)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close_time', 'Quote_asset_volume', 'Number_of_trades', 
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    print(f"Dados carregados: {data.shape[0]} linhas")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

# Parâmetros iniciais
initial_capital = 10000.0
stop_loss_diff = 150  # Defina o stop-loss
take_profit_diff = 150  # Defina o take-profit

# Carregar os dados históricos do BTC/USD com intervalo de 1 minuto
data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_15MINUTE, '7 days ago UTC')

# Calcular indicadores
data = calculate_indicators(data)

# Implementar a estratégia
data = implement_strategy(data, stop_loss_diff, take_profit_diff, initial_capital)

if data.empty:
    print("Erro: Nenhum dado foi carregado.")
else:
    print(f"Dados carregados: {data.shape[0]} linhas")

# Inicializar a visualização
plt.ion()  # Habilitar modo interativo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plotar as linhas iniciais
line_price, = ax1.plot([], [], label='Preço BTC/USD', color='blue', lw=2)
line_ema_5, = ax1.plot([], [], label='EMA 5', color='red', lw=1)
line_ema_21, = ax1.plot([], [], label='EMA 21', color='green', lw=1)
line_long_entries, = ax1.plot([], [], '^', markersize=10, color='lime', label='Entrada Longa')
line_long_exits, = ax1.plot([], [], 'v', markersize=10, color='darkgreen', label='Saída Longa')
line_short_entries, = ax1.plot([], [], 'v', markersize=10, color='darkred', label='Entrada Vendida')
line_short_exits, = ax1.plot([], [], '^', markersize=10, color='pink', label='Saída Vendida')
line_stop_loss, = ax1.plot([], [], 'x', markersize=10, color='red', label='Stop Loss')
line_take_profit, = ax1.plot([], [], 'x', markersize=10, color='green', label='Take Profit')

line_capital, = ax2.plot([], [], label='Evolução do Capital', color='purple', lw=2)

ax1.legend(loc='upper left')
ax1.grid()
ax2.legend(loc='upper left')
ax2.grid()

fig.autofmt_xdate()

# Função para verificar se as horas e minutos atuais são múltiplos de 5
def is_time_multiple_of_five():
    current_time = datetime.now()
    return current_time.minute % 5 == 0

# Variáveis para armazenar o último sinal enviado
last_buy_signal_time = None
last_sell_signal_time = None


def update_graph(frame):
    """Atualiza o gráfico com novos dados se a hora e os minutos forem múltiplos de 5."""
    global data, last_buy_signal_time, last_sell_signal_time

    try:
        # Verificar se a hora e os minutos são múltiplos de 5
        if not is_time_multiple_of_five():
            print("A hora e/ou os minutos atuais não são múltiplos de 5. O gráfico não será atualizado.")
            return  # Não atualizar o gráfico

        print("Atualizando o gráfico...")

        data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_5MINUTE, '7 days ago UTC')
        data = calculate_indicators(data)
        data = implement_strategy(data, stop_loss_diff, take_profit_diff, initial_capital)

        data = data.tail(500)  # Ajuste o número de pontos a serem plotados

        # Atualizar gráficos
        line_price.set_data(data.index, data['Close'])
        line_ema_5.set_data(data.index, data['EMA_5'])
        line_ema_21.set_data(data.index, data['EMA_21'])

        # Preparar os dados de sinais
        long_entries = data[(data['Trade'] == 1) & (data['Position'] == 1)]
        long_exits = data[(data['Trade'] == -1) & (data['Position'] == 0) & (data['Stop_Loss'].notnull() | data['Take_Profit'].notnull())]

        short_entries = data[(data['Trade'] == -1) & (data['Position'] == -1)]
        short_exits = data[(data['Trade'] == 1) & (data['Position'] == 0) & (data['Stop_Loss'].notnull() | data['Take_Profit'].notnull())]

        stop_loss_signals = data.dropna(subset=['Stop_Loss'])
        take_profit_signals = data.dropna(subset=['Take_Profit'])

        # Atualizar sinais no gráfico
        line_long_entries.set_data(long_entries.index, long_entries['Close'])
        line_long_exits.set_data(long_exits.index, long_exits['Close'])
        line_short_entries.set_data(short_entries.index, short_entries['Close'])
        line_short_exits.set_data(short_exits.index, short_exits['Close'])
        line_stop_loss.set_data(stop_loss_signals.index, stop_loss_signals['Stop_Loss'])
        line_take_profit.set_data(take_profit_signals.index, take_profit_signals['Take_Profit'])

        line_capital.set_data(data.index, data['Capital'])

        # Verificar e enviar sinais de entrada longa
        if not long_entries.empty and long_entries.index[-1] != last_buy_signal_time:
            last_buy_signal_time = long_entries.index[-1]
            buy_price = long_entries['Close'].iloc[-1]
            print(f"\nEntrada Longa em {last_buy_signal_time}: {buy_price}")
            send_telegram_message(f"Entrada Longa em {last_buy_signal_time}: {buy_price}")

        # Verificar e enviar sinais de saída longa
        elif not long_exits.empty and long_exits.index[-1] != last_sell_signal_time:
            last_sell_signal_time = long_exits.index[-1]
            sell_price = long_exits['Close'].iloc[-1]
            print(f"\nSaída Longa em {last_sell_signal_time}: {sell_price}")
            send_telegram_message(f"Saída Longa em {last_sell_signal_time}: {sell_price}")

        # Verificar e enviar sinais de entrada vendida
        elif not short_entries.empty and short_entries.index[-1] != last_sell_signal_time:
            last_sell_signal_time = short_entries.index[-1]
            sell_price = short_entries['Close'].iloc[-1]
            print(f"\nEntrada Vendida em {last_sell_signal_time}: {sell_price}")
            send_telegram_message(f"Entrada Vendida em {last_sell_signal_time}: {sell_price}")

        # Verificar e enviar sinais de saída vendida
        elif not short_exits.empty and short_exits.index[-1] != last_buy_signal_time:
            last_buy_signal_time = short_exits.index[-1]
            buy_price = short_exits['Close'].iloc[-1]
            print(f"\nSaída Vendida em {last_buy_signal_time}: {buy_price}")
            send_telegram_message(f"Saída Vendida em {last_buy_signal_time}: {buy_price}")

        # Ajustar limites dos eixos
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw()

    except Exception as e:
        print(f"Erro ao atualizar o gráfico: {e}")
        send_telegram_message(f"Erro ao atualizar o gráfico: {e}")
        time.sleep(10)  # Espera 10 segundos antes de tentar novamente

# Configurar animação
ani = FuncAnimation(fig, update_graph, interval=60000, cache_frame_data=False)  # Atualiza a cada 1 minuto (60000ms)

plt.ioff()  # Desabilitar modo interativo
plt.show()
