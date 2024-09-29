# Backtesting de Estratégias com Médias Móveis para BTC/USD utilizando a API da Binance

O mercado de criptomoedas é conhecido pela sua volatilidade, o que torna a análise técnica uma ferramenta poderosa para investidores que buscam explorar possíveis pontos de entrada e saída no mercado. Neste artigo, vou compartilhar um projeto de backtesting que desenvolvi, utilizando médias móveis (EMAs) e a API da Binance para encontrar possíveis oportunidades de investimento no par BTC/USD.

## Introdução

Backtesting é uma técnica essencial no desenvolvimento de estratégias de investimento, pois permite avaliar como uma estratégia teria se comportado no passado, utilizando dados históricos. Para este projeto, criei um script em Python que utiliza as EMAs para determinar pontos de compra e venda com base em dados do mercado BTC/USD.

Além disso, este projeto utiliza a API da Binance para coletar dados históricos e simular operações com capital fictício, tudo isso com a integração de sinais de compra e venda enviados via Telegram.

## Tecnologias Utilizadas

- **Python**: Para a lógica principal do backtest e manipulação de dados.
- **Binance API**: Para obter dados históricos do par BTC/USD.
- **Médias Móveis Exponenciais (EMAs)**: Para identificar tendências no mercado.
- **Telegram**: Para notificar sinais de compra e venda em tempo real.

## Funcionamento do Projeto

O projeto se baseia em três principais etapas:

### 1. Coleta de Dados Históricos

Utilizando a API da Binance, o script coleta dados históricos do par BTC/USD. Estes dados incluem informações como o preço de abertura, fechamento, volume negociado, entre outros.

```python
def get_historical_data(symbol, interval, period):
    klines = client.get_historical_klines(symbol, interval, period)
    data = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
```

### 2. Cálculo das EMAs e Sinais de Entrada/Saída

O cálculo das EMAs é feito para dois períodos diferentes, uma EMA de curto prazo e outra de longo prazo. A estratégia é simples: quando a EMA de curto prazo cruza acima da EMA de longo prazo, é gerado um sinal de compra. O contrário gera um sinal de venda.

```python
def calculate_indicators(data, ema_short_period, ema_long_period):
    data['ema_short'] = data['Close'].ewm(span=ema_short_period, adjust=False).mean()
    data['ema_long'] = data['Close'].ewm(span=ema_long_period, adjust=False).mean()
    data['signal'] = np.where(data['ema_short'] > data['ema_long'], 1.0, -1.0)
    return data
```

### 3. Backtest e Simulação de Portfólio

A última etapa envolve a execução do backtest. Aqui, o script simula as operações de compra e venda baseadas nos sinais gerados pelas EMAs, contabilizando o capital inicial e avaliando o desempenho da estratégia.

```python
def backtest(data, initial_capital):
    positions = data['signal'].fillna(0.0)
    cash_flow = -positions.diff().multiply(data['Close'])
    portfolio = pd.DataFrame({
        'positions': positions.multiply(data['Close']),
        'cash': initial_capital + cash_flow.cumsum()
    })
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    return portfolio
```

## Resultados

Após o backtest, a estratégia é avaliada para encontrar a melhor combinação de parâmetros das EMAs. Em meu teste, explorei diferentes valores para a EMA curta e longa, observando que a melhor combinação foi uma EMA curta de 3 períodos e uma EMA longa de 23 períodos.

Além disso, o script também envia sinais de compra e venda para o Telegram sempre que uma nova operação é identificada, facilitando o acompanhamento das operações em tempo real.

## Conclusão

Esse projeto demonstra como é possível utilizar técnicas de análise técnica e a API da Binance para testar estratégias de investimento no mercado de criptomoedas. Embora os resultados do backtesting forneçam uma visão sobre o desempenho passado de uma estratégia, é importante lembrar que resultados passados não garantem sucesso futuro.

Se você tiver interesse em explorar esse código mais a fundo, o repositório completo está disponível no meu GitHub.
