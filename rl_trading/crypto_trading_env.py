import numpy as np
import pandas as pd
from typing import Tuple

class CryptoTradingEnv:
    def __init__(self, dataset: pd.DataFrame, initial_balance: float, max_lose_percent: float):
        """
        Inicializa o ambiente de trading de criptomoedas.

        Args:
            dataset (pd.DataFrame): Dados históricos da criptomoeda.
            initial_balance (float): Saldo inicial em dinheiro.
            max_lose_percent (float): Percentual máximo de perda permitido antes de encerrar.
        """
        # Configuração inicial
        self.max_steps = 500
        self.data = dataset.reset_index(drop=True)  # Reseta o índice do DataFrame
        self.initial_balance = initial_balance
        self.max_lose_percent = max_lose_percent

        self.start = np.random.randint(0, (len(self.data)-1) - self.max_steps)
        self.end = self.start + self.max_steps
        self.dataset = self.data[self.start:self.end].reset_index(drop=True)

        # Variáveis do ambiente
        self.balance = initial_balance  # Saldo atual em dinheiro
        self.crypto_balance = 0.0       # Quantidade de criptomoedas em carteira
        self.current_step = 0           # Passo atual no dataset
        self.done = False               # Flag para indicar o fim do episódio
        self.starting_price = self.dataset.loc[0, 'Close']  # Preço inicial de referência
        self.last_action = None         # Última ação executada

    def reset(self):
        """
        Reseta o ambiente para o estado inicial.

        Returns:
            dict: Estado inicial do ambiente.
        """
        self.start = np.random.randint(0, (len(self.data)-1) - 500)
        self.end = self.start + 500
        self.dataset = self.data[self.start:self.end].reset_index(drop=True)
        self.balance = self.initial_balance
        self.crypto_balance = 0.0
        self.current_step = 0
        self.done = False
        self.starting_price = self.dataset.loc[0, 'Close']
        return self._get_current_state()

    def _get_current_state(self) -> pd.Series:
        """
        Retorna o estado atual do ambiente, baseado na etapa atual.

        Returns:
            pd.Series: Linha atual do dataset.
        """
        return self.dataset.iloc[self.current_step]

    def step(self, action: int) -> Tuple[pd.Series, float, bool]:
        """
        Executa uma ação e avança o ambiente.

        Args:
            action (int): Ação a ser executada (0: buy, 1: sell, 2: hold).

        Returns:
            Tuple[pd.Series, float, bool]: (Estado atual, lucro, flag de finalização).
        """
        if self.done:
            raise ValueError("O ambiente já terminou. Por favor, resete-o para um novo episódio.")

        # Obtém o preço atual de fechamento
        current_price = self.dataset.loc[self.current_step, 'Close']
        profit = 0.0

        if action == self.last_action:  # Ação: Hold
            profit = self._hold(current_price)


        elif action == 0:  # Ação: Buy
            profit = self._buy(current_price)

        elif action == 1:  # Ação: Sell
            profit = self._sell(current_price)

        
        else:
            raise ValueError(f"Ação inválida ({action}). Por favor, escolha entre 0 (buy), 1 (sell) ou repita a última ação para hold.")

        # Incrementa para o próximo passo
        if self.current_step < len(self.dataset) - 1:
            self.current_step += 1
        else:
            self.done = True

        # Verifica a condição de finalização
        self.done = self._check_done()

        # Retorna o estado atual, lucro e flag de finalização
        self.last_action = action
        return self._get_current_state(), profit, self.done

    def _buy(self, price: float): #  Action 0
        """
        Realiza a compra de criptomoedas, se houver saldo disponível.

        Args:
            price (float): Preço atual da criptomoeda.
        """
        if self.balance > 0:
            # Compra o máximo possível com o saldo disponível
            amount_to_buy = self.balance / price
            self.crypto_balance += amount_to_buy
            self.balance = 0  # Todo o dinheiro foi usado na compra
            profit = self.crypto_balance * price
            return profit - self.initial_balance
        return 0.0

    def _sell(self, price: float) -> float: # Action 1
        """
        Realiza a venda de criptomoedas, se houver saldo em criptomoeda.

        Args:
            price (float): Preço atual da criptomoeda.

        Returns:
            float: Lucro obtido com a venda.
        """
        if self.crypto_balance > 0:
            # Vende todas as criptomoedas
            profit = self.crypto_balance * price
            self.balance += profit
            self.crypto_balance = 0  # Todo o saldo de cripto foi vendido
            return profit - self.initial_balance
        return 0.0
    
    def _hold(self, price: float) -> float: # Action 2
        if self.crypto_balance > 0:
            profit = self.crypto_balance * price
        elif self.balance > 0:
            profit = self.balance
        else:
            profit = 0
        return profit - self.initial_balance

    def _check_done(self) -> bool:
        """
        Verifica se o ambiente deve ser encerrado com base no limite de perda.

        Returns:
            bool: True se o ambiente terminou, False caso contrário.
        """
        current_price = self.dataset.loc[self.current_step, 'Close']
        total_value = self.balance + (self.crypto_balance * current_price)
        # return total_value <= 0
        loss_percent = 1 - (total_value / self.initial_balance)
        return loss_percent > self.max_lose_percent
