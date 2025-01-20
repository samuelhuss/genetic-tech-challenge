import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


horas_trabalho = 8
semanas_no_mes = 4

# Funções
def gerar_individuo(num_funcionarios, dias_semana, horarios_possiveis):
    individuo = np.full((num_funcionarios, dias_semana), -1, dtype=int)
    for f in range(num_funcionarios):
        num_dias_trabalho = np.random.randint(max(1, dias_semana // 2 - 1), min(dias_semana + 1, dias_semana // 2 + 2))
        dias_trabalho_funcionario = random.sample(range(dias_semana), num_dias_trabalho)
        for dia in dias_trabalho_funcionario:
            individuo[f, dia] = np.random.choice(horarios_possiveis)
    return individuo

def calcular_max_desvio(caixas_desejados, dias_semana):
    return sum(caixas_desejados.values()) * dias_semana * max(caixas_desejados.values()) or 1

def calcular_fitness(individuo, caixas_desejados, horas_func, dias_semana, horas_trabalho, max_horas_semana, min_folgas_consecutivas, max_folgas_semana, max_desvio, peso_excesso_horas=0.1, peso_excesso_folgas=0.05):
    num_funcionarios = individuo.shape[0]
    cobertura = np.zeros((dias_semana, len(horas_func)), dtype=int)
    fitness = 1.0 

    for f in range(num_funcionarios):
        horas_trabalhadas_semana = 0
        folgas_consecutivas = 0
        folgas_na_semana = 0
        dias_trabalhados = 0 

        for d in range(dias_semana):
            start = individuo[f, d]
            if start != -1:
                horas_trabalhadas_semana += horas_trabalho
                folgas_consecutivas = 0
                cobertura[d, start - horas_func[0]:start - horas_func[0] + horas_trabalho] += 1
                dias_trabalhados += 1
            else:
                folgas_consecutivas += 1

        if folgas_consecutivas >= min_folgas_consecutivas and dias_trabalhados > 0: #Só desconta se trabalhou algum dia
            horas_trabalhadas_semana -= (folgas_consecutivas // min_folgas_consecutivas)
        folgas_na_semana = dias_semana - dias_trabalhados

        if horas_trabalhadas_semana > max_horas_semana:
            excesso_horas = horas_trabalhadas_semana - max_horas_semana
            penalidade_horas = excesso_horas * peso_excesso_horas
            fitness *= max(0, 1 - penalidade_horas)
        

        if folgas_na_semana > max_folgas_semana:
            excesso_folgas = folgas_na_semana - max_folgas_semana
            penalidade_folgas = excesso_folgas * peso_excesso_folgas
            fitness *= max(0, 1 - penalidade_folgas)

    soma_desvios = np.sum((cobertura - np.array(list(caixas_desejados.values())))**2)
    fitness_demanda = 1 - (soma_desvios / max_desvio)

    peso_demanda = 0.95
    fitness *= fitness_demanda * peso_demanda
    return fitness
    
def selecao_torneio(populacao, fitnesses, tamanho_torneio=5):
    candidatos = random.sample(range(len(populacao)), tamanho_torneio)
    melhor = candidatos[0]
    for c in candidatos[1:]:
        if fitnesses[c] > fitnesses[melhor]:
            melhor = c
    return populacao[melhor]

def cruzamento_uniforme(pai1, pai2):
    num_funcionarios, dias_semana = pai1.shape
    filho = np.full((num_funcionarios, dias_semana), -1, dtype=int)
    for f in range(num_funcionarios):
        for d in range(dias_semana):
            if random.random() < 0.5:
                filho[f, d] = pai1[f, d]
            else:
                filho[f, d] = pai2[f, d]
    return filho

def mutacao(individuo, taxa, horarios_possiveis):
    num_funcionarios, dias_semana = individuo.shape
    for f in range(num_funcionarios):
        for d in range(dias_semana):
            if np.random.rand() < taxa:
                opcoes = list(horarios_possiveis) + [-1]
                individuo[f, d] = np.random.choice(opcoes)
    return individuo

def elitismo(populacao, fitnesses, tamanho_elite=2):
    elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:tamanho_elite]
    elite_individuos = [populacao[i] for i in elite_indices]
    return elite_individuos

# Configurações iniciais da página
st.set_page_config(
    page_title="Escala de Funcionários com Algoritmo Genético",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
            ## Sobre este Aplicativo

            Este aplicativo utiliza um algoritmo genético para otimizar a escala de funcionários.

            **Funcionalidades:**

            *   Geração automática de escalas.
            *   Visualização da escala em tabela.
            *   Download da escala em CSV.
            *   Gráficos comparativos.
        """
    }
)

st.markdown("<h1 style='text-align: center; color: #333;'>Escala de Funcionários com Algoritmo Genético</h1>", unsafe_allow_html=True)

# Sidebar para parâmetros
with st.sidebar:
    st.header("Parâmetros do Algoritmo")
    num_funcionarios = st.slider("Número de Funcionários", 5, 30, 15)
    pop_size = st.slider("Tamanho da População", 20, 200, 100)
    geracoes = st.slider("Número de Gerações", 100, 5000, 1500)
    taxa_mutacao_inicial = st.slider("Taxa de Mutação Inicial", 0.01, 0.5, 0.2)
    tamanho_torneio = st.slider("Tamanho do Torneio", 2, 10, 5)
    tamanho_elite = st.slider("Tamanho da Elite", 1, 10, 2)
    max_horas_semana = st.slider("Máximo de horas por semana", 1, 40, 40)
    min_folgas_consecutivas = st.slider("Mínimo de folgas consecutivas", 0, 7, 1)
    max_folgas_semana = st.slider("Máximo de folgas na semana", 0, 4, 2)
    dias_semana = st.slider("Dias da Semana", 1, 7, 7)
    horarios_possiveis_min = st.slider("Horário inicial mínimo", 0, 23, 7)
    horarios_possiveis_max = st.slider("Horário inicial máximo", 0, 23, 14)

    st.header("Demanda por Horário")
    horas_func = list(range(7, 22))
    caixas_desejados = {}
    for hora in horas_func:
        caixas_desejados[hora] = st.number_input(f"Funcionários às {hora}:00", min_value=0, step=1, value=4 if hora == 7 else 5 if hora == 8 else 6 if hora == 9 else 9 if hora == 10 else 10 if 10 < hora < 14 else 9 if 13 < hora < 16 else 7 if 15 < hora < 18 else 10 if 17 < hora < 20 else 7 if 19 < hora < 21 else 4)

horarios_possiveis = range(horarios_possiveis_min, horarios_possiveis_max + 1)
dias_no_mes = dias_semana * semanas_no_mes

if st.button("Gerar Escala"):
    with st.spinner("Gerando escala..."):
        populacao = [gerar_individuo(num_funcionarios, dias_semana, horarios_possiveis) for _ in range(pop_size)]
        max_desvio = calcular_max_desvio(caixas_desejados, dias_semana) 
        melhores_fitness = []
        fitness_medios = []

        for g in range(geracoes):
            fitnesses = [calcular_fitness(individuo, caixas_desejados, horas_func, dias_semana, horas_trabalho, max_horas_semana, min_folgas_consecutivas, max_folgas_semana, max_desvio) for individuo in populacao] 
            melhor_fitness = max(fitnesses)
            media_fitness = sum(fitnesses) / len(fitnesses)
            melhores_fitness.append(melhor_fitness)
            fitness_medios.append(media_fitness)

            if g % (geracoes // 10 if geracoes > 10 else 1) == 0:  # Mostrar progresso a cada 10%
                st.write(f"Geração {g}: Melhor Fitness = {melhor_fitness:.4f}, Fitness Médio = {media_fitness:.4f}")

            elite = elitismo(populacao, fitnesses, tamanho_elite)
            nova_populacao = elite
            while len(nova_populacao) < pop_size:
                pai1 = selecao_torneio(populacao, fitnesses, tamanho_torneio)
                pai2 = selecao_torneio(populacao, fitnesses, tamanho_torneio)
                filho1 = cruzamento_uniforme(pai1, pai2)
                filho2 = cruzamento_uniforme(pai2, pai1)
                taxa_mutacao = taxa_mutacao_inicial * (1 - g / geracoes)
                filho1 = mutacao(filho1, taxa_mutacao, horarios_possiveis)
                filho2 = mutacao(filho2, taxa_mutacao, horarios_possiveis)
                nova_populacao.extend([filho1, filho2])
            populacao = nova_populacao[:pop_size]

        fitnesses = [calcular_fitness(individuo, caixas_desejados, horas_func, dias_semana, horas_trabalho, max_horas_semana, min_folgas_consecutivas, max_folgas_semana, max_desvio) for individuo in populacao]
        melhor = populacao[fitnesses.index(max(fitnesses))]

        st.success(f"Escala gerada com sucesso!") # Mensagem de sucesso
         # Criando calendário mensal (DataFrame)
        calendario_mensal = []
        for f in range(num_funcionarios):
            escala_mensal = []
            for w in range(semanas_no_mes):
                escala_mensal.extend(melhor[f])
            calendario_mensal.append(escala_mensal)

        dias_label = [f"Dia {i+1}" for i in range(dias_no_mes)]
        linhas = []
        for f in range(num_funcionarios):
            linha = [f"Func {f+1}"]
            for d in range(dias_no_mes):
                horario = calendario_mensal[f][d]
                if horario == -1:
                    linha.append("Folga")
                else:
                    linha.append(f"{horario}:00")
            linhas.append(linha)

        colunas = ["Funcionário"] + dias_label
        df = pd.DataFrame(linhas, columns=colunas)

        st.subheader("Escala Mensal") # Subtítulo
        st.dataframe(df)

        @st.cache_data # Mantém o cache para o download ser mais rápido
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='calendario_mensal.csv',
            mime='text/csv',
        )

        caixas_por_hora = [{h: 0 for h in horas_func} for _ in range(dias_semana)]
        for f in range(num_funcionarios):
            for d in range(dias_semana):
                start = melhor[f][d]
                if start != -1:
                    for h_offset in range(horas_trabalho):
                        h = start + h_offset
                        if h in caixas_por_hora[d]:
                            caixas_por_hora[d][h] += 1

        cobertura_real = {h: 0 for h in horas_func}
        for d in range(dias_semana):
            for h in horas_func:
                cobertura_real[h] += caixas_por_hora[d][h]


        st.subheader("Distribuição de Funcionários")
        sns.set_theme(style="whitegrid")

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for d, caixas in enumerate(caixas_por_hora):
            horas_dia = list(caixas.keys())
            valores = list(caixas.values())
            sns.lineplot(x=horas_dia, y=valores, label=f"Dia {d+1}", marker='o', ax=ax1) 
        ax1.set_title("Distribuição de Funcionários Reais por Dia")
        ax1.set_xlabel("Hora")
        ax1.set_ylabel("Número de Funcionários")
        ax1.set_xticks(horas_func) #define os ticks do eixo x
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 7))

        window = 5
        melhores_fitness_suavizado = pd.Series(melhores_fitness).rolling(window=window, min_periods=1).mean()
        fitness_medios_suavizado = pd.Series(fitness_medios).rolling(window=window, min_periods=1).mean()

        sns.lineplot(x=range(len(melhores_fitness)), y=melhores_fitness_suavizado, label="Melhor Fitness (Suavizado)", marker='o', markersize=4, ax=ax2, color='green', linewidth=2)
        sns.lineplot(x=range(len(fitness_medios)), y=fitness_medios_suavizado, label="Fitness Médio (Suavizado)", linestyle="--", marker='o', markersize=4, ax=ax2, color='blue', linewidth=2)

        melhor_fitness_global = max(melhores_fitness)
        ax2.axhline(y=melhor_fitness_global, color='red', linestyle=':', label=f"Melhor Fitness Global: {melhor_fitness_global:.4f}")

        ax2.set_title("Evolução do Fitness", fontsize=16)
        ax2.set_xlabel("Gerações", fontsize=14)
        ax2.set_ylabel("Fitness", fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

        min_y = min(min(melhores_fitness), min(fitness_medios))
        max_y = max(max(melhores_fitness), max(fitness_medios))
        ax2.set_ylim(min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1)

        st.pyplot(fig2)

        demanda_semanal = {h: caixas_desejados[h] * 7 for h in horas_func}
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=horas_func, y=list(demanda_semanal.values()), label="Demanda Mensal", marker='o', ax=ax3)
        sns.lineplot(x=horas_func, y=list(cobertura_real.values()), label="Cobertura Real", marker='o', ax=ax3)
        ax3.set_title("Comparativo de Demanda Mensal e Cobertura Real")
        ax3.set_xlabel("Hora")
        ax3.set_ylabel("Número de Funcionários")
        ax3.legend()
        st.pyplot(fig3)
