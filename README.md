# Especificações

Contexto: **Aprendizagem por Reforço**
Linguagem: **Python**
Problema: Preciso **criar um ambiente** de trading de cripto moedas **para treinamento** de um modelo de aprendizagem por reforço.
Regras:

    1. Esse ambiente inicia recebendo um initial_balance, um max_lose_percent e um dataset com as seguintes informações sobre a cripto moeda (Index=Date, Open, High, Low, Close, Volume, m_avg_7, m_avg_25, m_avg_99, close_diff) Sendo m_avg_7, m_avg_25, m_avg_99 a média móvel de 7, 25 e 99 dias, respectivamente e close_diff o resultado da função pct_change() na coluna Close.
    2. Esse ambiente deve receber um valor inicial (initial_balance), para realizar o trading
    3. Esse ambiente deve receber 3 possíveis ações, buy, sell, hold.
    4. A ação de buy só pode ser executada se houver dinheiro em caixa e pode ser feita uma compra parcial, ou seja, de frações do valor da moeda
    5. A ação de sell só pode ser executada se houver algum valor em cripto moeda
    6. A ação de hold pode ser executada em qualquer momento, seja para aguardar o momento mais oportuno para realizar a compra, ou para aguardar o valor ótimo pra venda
    7. A cada step o ambiente deve receber uma ação em formato numérico 0: buy, 1: sell, 2: hold, deve retornar os seguintes dados (linha do dataset da data em questão, profits, done). Sendo que a variável done deve validar a cada venda se (1 - (o valor atual dividido pelo valor inicial )) > max_lose_percent