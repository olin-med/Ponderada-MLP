import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RedeNeural:
    """
    Rede Neural com uma camada oculta utilizando gradiente descendente e backpropagation.
    """
    def __init__(self, dados_entrada, rotulos, taxa_aprendizado=0.1, iteracoes=100, tamanho_entrada=2, tamanho_oculto=2, tamanho_saida=1):
        self.dados_entrada = dados_entrada
        self.rotulos = rotulos
        self.taxa_aprendizado = taxa_aprendizado
        self.iteracoes = iteracoes
        self.pesos_entrada_para_oculto = np.random.uniform(size=(tamanho_entrada, tamanho_oculto))
        self.pesos_oculto_para_saida = np.random.uniform(size=(tamanho_oculto, tamanho_saida))
        self.vieses_oculto = np.random.uniform(size=(1, tamanho_oculto))
        self.vieses_saida = np.random.uniform(size=(1, tamanho_saida))
        self.historico_erro = []

    def passagem_para_frente(self, lote):
        self.oculta = np.dot(lote, self.pesos_entrada_para_oculto) + self.vieses_oculto
        self.saida_oculta = self.sigmoid(self.oculta)
        self.saida = np.dot(self.saida_oculta, self.pesos_oculto_para_saida) + self.vieses_saida
        self.saida_final = self.sigmoid(self.saida)
        return self.saida_final

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return x * (1 - x)

    def calcular_custo(self, saida, rotulos):
        # Cálculo do erro quadrático médio
        return np.mean((saida - rotulos) ** 2)

    def ajustar_pesos(self):
        # Backpropagation
        delta_saida = (self.rotulos - self.saida_final) * self.derivada_sigmoid(self.saida_final)
        erro_oculto = delta_saida.dot(self.pesos_oculto_para_saida.T) * self.derivada_sigmoid(self.saida_oculta)
        
        gradiente_oculto_para_saida = self.saida_oculta.T.dot(delta_saida)
        gradiente_entrada_para_oculto = self.dados_entrada.T.dot(erro_oculto)
        
        self.pesos_oculto_para_saida += self.taxa_aprendizado * gradiente_oculto_para_saida
        self.pesos_entrada_para_oculto += self.taxa_aprendizado * gradiente_entrada_para_oculto
        
        self.vieses_saida += np.sum(self.taxa_aprendizado * delta_saida, axis=0, keepdims=True)
        self.vieses_oculto += np.sum(self.taxa_aprendizado * erro_oculto, axis=0, keepdims=True)

    def treinar_rede(self):
        for _ in range(self.iteracoes):
            saida = self.passagem_para_frente(self.dados_entrada)
            custo = self.calcular_custo(saida, self.rotulos)
            self.historico_erro.append(custo)
            self.ajustar_pesos()
        print("Treinamento concluído. Erro final: ", custo)

    def prever(self, ponto_dados):
        resultado = self.passagem_para_frente(ponto_dados)
        return 1 if resultado >= 0.5 else 0

    def testar_rede(self):
        resultados = []
        for dado in self.dados_entrada:
            resultados.append(self.prever(dado))
        return resultados

    def plotar_erro(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.historico_erro, label='Erro ao longo das iterações')
        plt.title('Erro ao longo das iterações de treinamento')
        plt.xlabel('Iterações')
        plt.ylabel('Erro')
        plt.legend()
        plt.show()

# Dados de treinamento
dados_entrada = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])

rotulos = np.array([
    [0],
    [1],
    [1],
    [0]])

rede_neural = RedeNeural(dados_entrada, rotulos, 0.2, 5000)
rede_neural.treinar_rede()
print("Resultados dos Testes XOR:", rede_neural.testar_rede())
rede_neural.plotar_erro()