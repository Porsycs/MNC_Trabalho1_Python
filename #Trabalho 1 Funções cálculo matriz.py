#Trabalho 1 Funções cálculo matriz
import numpy as np

def solicitar_matriz():
    try:
        
        ordem = int(input("Digite a ordem da matriz: "))
        
        linhas = colunas = ordem

        matriz = np.zeros((linhas, colunas))

        for i in range(linhas):
            for j in range(colunas):
                matriz[i, j] = float(input(f"Digite o elemento da posição ({i+1},{j+1}): "))
                    
        return matriz, ordem
    except Exception as e:
        raise Exception("Erro na criação da matriz. Verifique os valores digitados.")
    
def vetor_independente(ordem):
        vetor_termos_independentes = np.zeros(ordem)
        for i in range(ordem):
            vetor_termos_independentes[i] = float(input(f"Digite o termo {i+1}: "))
        return vetor_termos_independentes

def calculo_determinante(matriz, ordem):
    try:

        if ordem == 1:
            return matriz[0, 0]

        if ordem == 2:
            return matriz[0, 0] * matriz[1, 1] - matriz[0, 1] * matriz[1, 0]

        det = 0
        for j in range(ordem):
            det += ((-1) ** j) * matriz[0, j] * calculo_determinante(np.delete(np.delete(matriz, 0, 0), j, 1))
        if np.isnan(det).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return det
    except Exception:
        raise Exception("Ocorreu um erro ao calcular a determinante.")

def cholesky(matriz, ordem):
    try:

        # Verifica se a matriz é positiva
        if not np.all(np.linalg.eigvals(matriz) > 0):
            raise ValueError("A matriz dos coeficientes não é positiva definida")

        vetor_termos_independentes = vetor_independente(ordem)

        L = np.zeros((ordem, ordem))
        
        # Decomposição de Cholesky
        for i in range(ordem):
            for j in range(i+1):
                if i == j:
                    L[i,i] = np.sqrt(matriz[i,i] - np.sum(L[i,:]**2))
                else:
                    L[i,j] = (matriz[i,j] - np.sum(L[i,:j] * L[j,:j])) / L[j,j]

        # Resolve LY = B
        y = np.zeros(ordem)
        for i in range(ordem):
            y[i] = vetor_termos_independentes[i] - np.dot(L[i,:i], y[:i])

        # Resolve UX = Y
        solucao = np.zeros(ordem)
        for i in range(ordem - 1, -1, -1):
            solucao[i] = (y[i] - np.dot(L[i+1:,i], solucao[i+1:])) / L[i,i]
        if np.isnan(solucao).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return solucao
    except Exception as e:
        raise Exception(e)


def gauss_compacto(matriz, ordem):
    try:
        
        vetor_termos_independentes = vetor_independente(ordem)
        
        # Matriz Ampliada
        matriz_aumentada = np.column_stack((matriz, vetor_termos_independentes))

        # Eliminação Gaussiana
        for i in range(ordem):
            pivo = matriz_aumentada[i, i]
            if(pivo == 0):
                raise Exception("Pivo zero, método não válido.")
            matriz_aumentada[i] /= pivo
            for j in range(i + 1, ordem):
                multiplicador = matriz_aumentada[j, i]
                matriz_aumentada[j] -= multiplicador * matriz_aumentada[i]
 
        solucao = np.zeros(ordem)
        for i in range(ordem - 1, -1, -1):
            solucao[i] = matriz_aumentada[i, -1] - np.dot(matriz_aumentada[i, i+1:-1], solucao[i+1:])        
        if np.isnan(solucao).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return solucao
    except Exception as e:
        raise Exception(e)

def gauss_jordan(matriz, ordem):
    try:
        
        vetor_termos_independentes = vetor_independente(ordem)
        # Construir a matriz aumentada
        matriz_aumentada = np.column_stack((matriz, vetor_termos_independentes))

        # Fase de eliminação
        for i in range(ordem):
            pivo = matriz_aumentada[i, i]
            if pivo == 0:
                raise ValueError("Pivo zero. Método não pode ser aplicado.")
            matriz_aumentada[i] /= pivo
            for j in range(ordem):
                if i != j:
                    multiplicador = matriz_aumentada[j, i]
                    matriz_aumentada[j] -= multiplicador * matriz_aumentada[i]
        if np.isnan(matriz_aumentada).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return matriz_aumentada[:, -1]
    except Exception as e:
        raise Exception(e)

def gauss_seidel(matriz, ordem):
    try:
        
        vetor_termos_independentes = vetor_independente(ordem)
        
        aproximacao_inicial = np.zeros(ordem)
        
        precisao = float(input("Digite a precisão: "))
        
        max_iteracoes = int(input("Digite o número máximo de iterações: "))
        
        x = np.copy(aproximacao_inicial)
        iteracoes = 0
        erro = float('inf')
    
        while erro > precisao and iteracoes < max_iteracoes:
            x_anterior = np.copy(x)
            for i in range(ordem):
                soma = np.dot(matriz[i, :i], x[:i]) + np.dot(matriz[i, i+1:], x_anterior[i+1:])
                x[i] = (vetor_termos_independentes[i] - soma) / matriz[i, i]

            erro = np.linalg.norm(x - x_anterior, np.inf)
            iteracoes += 1
        if np.isnan(x).any():
            raise ValueError("A solução retornou uma resposta inválida.")       
        return x, iteracoes
    except Exception as e:
        raise Exception(e)
    
def jacobi(matriz, ordem):
    try:
        vetor_termos_independentes = vetor_independente(ordem)
        
        aproximacao_inicial = np.zeros(ordem)
        
        precisao = float(input("Digite a precisão: "))
        
        max_iteracoes = int(input("Digite o número máximo de iterações: "))
        
        x = np.copy(aproximacao_inicial)
        iteracoes = 0
        erro = float('inf')
    
        while erro > precisao and iteracoes < max_iteracoes:
            x_anterior = np.copy(x)
            for i in range(ordem):
                soma = np.dot(matriz[i, :], x_anterior) - matriz[i, i] * x_anterior[i]
                x[i] = (vetor_termos_independentes[i] - soma) / matriz[i, i]

            erro = np.linalg.norm(x - x_anterior, np.inf)
            iteracoes += 1
        if np.isnan(x).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return x, iteracoes
    except Exception as e:
        raise Exception(e)

def matriz_inversa(matriz, ordem):
    try:
        
        # Construir a matriz aumentada
        matriz_aumentada = np.column_stack((matriz, np.eye(ordem)))

        # Fase de eliminação
        for i in range(ordem):
            pivo = matriz_aumentada[i, i]
            if pivo == 0:
                raise ValueError("Matriz singular, não é possível calcular a inversa.")
            matriz_aumentada[i] /= pivo
            for j in range(ordem):
                if i != j:
                    multiplicador = matriz_aumentada[j, i]
                    matriz_aumentada[j] -= multiplicador * matriz_aumentada[i]

        # Extrair a matriz inversa
        inversa = matriz_aumentada[:, ordem:]
        if np.isnan(inversa).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return inversa
    except Exception as e:
        raise Exception(e)

def triangular_inferior(matriz, ordem):
    try:
        print("Digite os termos independentes:")
        vetor_termos_independentes = vetor_independente(ordem)

        solucao = np.zeros(ordem)
        for i in range(ordem):
            soma = np.dot(matriz[i][:i], solucao[:i])
            solucao[i] = (vetor_termos_independentes[i] - soma) / matriz[i][i]
        if np.isnan(solucao).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return solucao
    except Exception:
        raise Exception("Erro ao calcular a matriz triangular inferior.")

def triangular_superior(matriz, ordem):
    try:
        print("Digite os termos independentes:")
        vetor_termos_independentes = vetor_independente(ordem)
            
        solucao = np.zeros(ordem)
        for i in range(ordem - 1, -1, -1):
            soma = np.dot(matriz[i][i+1:], solucao[i+1:])
            solucao[i] = (vetor_termos_independentes[i] - soma) / matriz[i][i]
        if np.isnan(solucao).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return solucao
    
    except Exception:
        raise Exception("Erro ao calcular a matriz triangular superior.")
    
def decomposicao_LU(matriz, ordem):
    try:
        
        print("Digite os termos independentes:")
        vetor_termos_independentes = vetor_independente(ordem)
        
        L = np.eye(ordem)
        U = np.zeros((ordem, ordem))

        for k in range(ordem):
            # Matriz U
            for j in range(k, ordem):
                soma = sum(L[k][s] * U[s][j] for s in range(k))
                U[k][j] = matriz[k][j] - soma

            # Matriz L
            for i in range(k+1, ordem):
                soma = sum(L[i][s] * U[s][k] for s in range(k))
                L[i][k] = (matriz[i][k] - soma) / U[k][k]

        # Resolve LY = B
        y = np.zeros(ordem)
        for i in range(ordem):
            y[i] = vetor_termos_independentes[i] - sum(L[i][j] * y[j] for j in range(i))

        # Resolve UX = Y
        x = np.zeros(ordem)
        for i in range(ordem - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, ordem))) / U[i][i]
        if np.isnan(x).any():
            raise ValueError("A solução retornou uma resposta inválida.")
        return x
    except Exception:
        raise Exception("Erro ao calcular a decomposição LU.")

# Menu para escolher a função
def menu():
    try:
        print("Escolha uma das funções:")
        print("0. Sair")
        print("1. Calculo Determinante")
        print("2. Triangular Inferior")
        print("3. Triangular Superior")
        print("4. Decomposição LU")
        print("5. Cholesky")
        print("6. Gauss Compacto")
        print("7. Gauss Jordan")
        print("8. Jacobi")
        print("9. Gauss Seidel")
        print("10. Matriz Inversa")

        opcao = int(input("Digite o número da função desejada: "))
        return opcao
    except Exception:
        raise Exception("Erro ao escolher a função. Verifique o valor digitado...")

while True:
    opcao = menu()
    if opcao > 0 and opcao <= 10:
        matriz, ordem = solicitar_matriz()
    if opcao == 0:
        print("Programa encerrado.")
        break
    if opcao == 1:
        try:
            D = calculo_determinante(matriz, ordem)
            print("Determinante: ", D)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 2:
        try:
            V = triangular_inferior(matriz, ordem)
            print("Vetor Solução: ", V)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 3:
        try:
            V = triangular_superior(matriz, ordem)
            print("Vetor Solução: ", V)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 4:
        try:
            V = decomposicao_LU(matriz, ordem)
            print("Vetor Solução: ", V)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 5:
        try:
            V = cholesky(matriz, ordem)
            print("Vetor Solução: ", V)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 6:
        try:
            V = gauss_compacto(matriz, ordem)
            print("Vetor Solução: ", V)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 7:
        try:
            V = gauss_jordan(matriz, ordem)
            print("Vetor Solução: ", V)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 8:
        try:
            V, I = jacobi(matriz, ordem)
            print("Vetor Solução: ", V)
            print("Número de iterações: ", I)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 9:
        try:
            V, I = gauss_seidel(matriz, ordem)
            print("Vetor Solução: ", V)
            print("Número de iterações: ", I)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 10:
        try:
            I = matriz_inversa(matriz, ordem)
            print("Matriz Inversa:\n", I)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    else:
        input("Função não encontrada. Pressione enter para digitar novamente...")
        
    