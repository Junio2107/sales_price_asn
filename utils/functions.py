
import statsmodels.formula.api as smf
import pandas as pd

def forward_selection(data, response, significance_level=0.05):
    """
    data               : DataFrame contendo a variável resposta e todas as candidatas (X).
    response           : string com o nome da variável dependente (y).
    significance_level : valor de corte para p-valor na inclusão de cada variável.
    
    Retorna:
    - model: o modelo final ajustado (objeto statsmodels RegressionResults).
    - selected_vars: lista das variáveis selecionadas.
    """

    # Separamos as colunas exceto a de resposta
    remaining_vars = set(data.columns)
    remaining_vars.remove(response)
    
    # Começamos sem nenhuma variável
    selected_vars = []
    
    # Enquanto estiver encontrando variáveis com p-valor significativo, continua
    changed = True
    
    while changed and len(remaining_vars) > 0:
        changed = False
        
        # Para cada variável ainda não selecionada, testamos adicioná-la no modelo
        pvals = []
        for candidate in remaining_vars:
            # Montamos a fórmula, que é "response ~ variaveis_selecionadas + candidato"
            formula = f"{response} ~ {' + '.join(selected_vars + [candidate])}" if selected_vars \
                      else f"{response} ~ {candidate}"
            
            # Ajustamos o modelo
            model = smf.ols(formula, data=data).fit()
            
            # Coletamos o p-valor do candidato
            # (o nome do parâmetro no pvalues será igual ao nome da variável candidata 
            #  ou 'Intercept' quando for o intercepto)
            pvals.append((candidate, model.pvalues[candidate]))
        
        # Ordenamos as variáveis candidatas por menor p-valor
        pvals.sort(key=lambda x: x[1])
        
        # Pegamos a candidata com menor p-valor
        best_candidate, best_pval = pvals[0]
        
        # Se o melhor p-valor for menor que o nível de significância, incluímos a variável
        if best_pval < significance_level:
            selected_vars.append(best_candidate)
            remaining_vars.remove(best_candidate)
            changed = True
    
    # Ao final, ajustamos o modelo definitivo com as variáveis selecionadas
    formula = f"{response} ~ {' + '.join(selected_vars)}" if selected_vars \
              else f"{response} ~ 1"  # Se nenhuma variável foi selecionada
    final_model = smf.ols(formula, data=data).fit()
    
    return final_model, selected_vars



def backward_selection_aic(data, response):
    """
    Realiza seleção retrógrada (backward elimination) baseada em AIC.
    
    Parâmetros
    ----------
    data : pandas.DataFrame
        DataFrame contendo a variável resposta e todas as variáveis preditoras.
    response : str
        Nome da coluna (variável) que será a resposta (y).
    
    Retorno
    -------
    final_model : statsmodels RegressionResults
        O modelo final, após a eliminação retrógrada.
    selected_vars : list
        Lista com o nome das variáveis que permaneceram no modelo final.
    """
    # 1. Definir o conjunto inicial de variáveis (todas exceto a resposta)
    remaining_vars = list(data.columns)
    remaining_vars.remove(response)
    
    # 2. Montar a fórmula completa: response ~ var1 + var2 + ...
    formula = f"{response} ~ {' + '.join(remaining_vars)}"
    current_model = smf.ols(formula, data=data).fit()
    
    # Critério atual (AIC do modelo cheio)
    current_aic = current_model.aic
    
    # 3. Loop para tentar remover variáveis que melhorem (reduzam) o AIC
    while True:
        aic_values = []
        removed_candidate = None
        
        # Testa remover cada variável (uma por vez) e calcula novo AIC
        for var in remaining_vars:
            vars_minus_one = [x for x in remaining_vars if x != var]
            formula_test = f"{response} ~ {' + '.join(vars_minus_one)}"
            
            model_test = smf.ols(formula_test, data=data).fit()
            aic_values.append((var, model_test.aic, model_test))
        
        # Ordena pelos menores valores de AIC (melhor)
        aic_values.sort(key=lambda x: x[1])
        best_candidate, best_aic, best_model = aic_values[0]
        
        # Se o melhor AIC encontrado for menor que o AIC atual, remove a variável
        if best_aic < current_aic:
            remaining_vars.remove(best_candidate)
            current_aic = best_aic
            current_model = best_model
        else:
            # Se não melhorou, paramos o loop
            break
    
    # 4. Retorna o modelo final e as variáveis escolhidas
    return current_model, remaining_vars



def stepwise_selection_both(data, response, 
                            initial_list=None, 
                            threshold_in=0.01, 
                            threshold_out=0.05, 
                            verbose=True):
    """
    Função para realizar Stepwise (both) usando p-valores e baseando-se no AIC.
    O procedimento avalia tanto a inclusão quanto a exclusão de variáveis
    a cada iteração.
    
    Parâmetros
    ----------
    data : pandas.DataFrame
        DataFrame com a variável resposta e todas as colunas candidatas.
    response : str
        Nome da variável resposta (y).
    initial_list : list ou None
        Lista inicial de variáveis para começar o modelo. Se None, começa sem variáveis.
    threshold_in : float
        p-valor limite para incluir uma variável.
    threshold_out : float
        p-valor limite para remover uma variável.
    verbose : bool
        Se True, imprime o log das etapas.
        
    Retorna
    -------
    final_model : statsmodels RegressionResults
        O modelo final (ajustado com statsmodels).
    selected_vars : list
        Lista final de variáveis selecionadas.
    """
    
    # Se nao for fornecida uma lista inicial, começamos sem nenhuma variável
    if initial_list is None:
        selected_vars = []
    else:
        # Fazemos uma cópia para não afetar a lista original
        selected_vars = list(initial_list)
    
    # Variáveis candidatas = todas exceto a resposta e as que já estão dentro
    remaining_vars = list(set(data.columns) - set(selected_vars) - {response})
    
    # Ajustamos um modelo inicial (pode ser intercepto se selected_vars estiver vazio)
    formula = f"{response} ~ {' + '.join(selected_vars)}" if selected_vars else f"{response} ~ 1"
    best_model = smf.ols(formula, data=data).fit()
    
    # AIC do modelo atual
    best_aic = best_model.aic
    improved = True
    
    while improved:
        improved = False
        
        # =============== PASSO 1: TENTAR INCLUIR ALGUMA VARIÁVEL ===============
        # Testamos adicionar cada variável que ainda não está no modelo
        candidates_for_inclusion = []
        for var in remaining_vars:
            formula_test = f"{response} ~ {' + '.join(selected_vars + [var])}"
            model_test = smf.ols(formula_test, data=data).fit()
            aic_test = model_test.aic
            pval_test = model_test.pvalues[var]
            candidates_for_inclusion.append((var, aic_test, pval_test, model_test))
        
        # Verificamos a melhor inclusão (menor AIC) dentre as candidatas
        candidates_for_inclusion.sort(key=lambda x: x[1])  # ordena por AIC (ascendente)
        
        if candidates_for_inclusion:
            best_candidate_inclusion, best_aic_inclusion, best_pval_inclusion, best_model_inclusion = candidates_for_inclusion[0]
        else:
            best_candidate_inclusion = None
            best_aic_inclusion = float('inf')
            best_pval_inclusion = 1.0
        
        # =============== PASSO 2: TENTAR REMOVER ALGUMA VARIÁVEL ===============
        # Testamos remover cada variável que já está no modelo
        candidates_for_removal = []
        if selected_vars:
            for var in selected_vars:
                vars_minus_one = [x for x in selected_vars if x != var]
                if len(vars_minus_one) == 0:
                    # Se removermos a última variável, modelo vira intercepto
                    formula_test = f"{response} ~ 1"
                else:
                    formula_test = f"{response} ~ {' + '.join(vars_minus_one)}"
                model_test = smf.ols(formula_test, data=data).fit()
                aic_test = model_test.aic
                # Para avaliar remoção, olhamos p-valor do var no modelo anterior?
                # ou com a var removida?
                # -> Normalmente se checa p-value do var no "modelo atual",
                #    mas aqui poderemos seguir a lógica: se AIC melhora, remove.
                
                candidates_for_removal.append((var, aic_test, model_test))
            
            # Ordenamos por AIC
            candidates_for_removal.sort(key=lambda x: x[1])  # menor AIC é melhor
            best_candidate_removal, best_aic_removal, best_model_removal = candidates_for_removal[0]
        else:
            # Se não há variáveis selecionadas, não podemos remover
            best_candidate_removal = None
            best_aic_removal = float('inf')
        
        # =============== DECIDIR MELHOR MUDANÇA (ADD ou REMOVE) ===============
        # O "both" do R observa p-valores e AIC simultaneamente. Uma prática comum:
        # 1. Se a melhor inclusão diminui o AIC e p-valor < threshold_in, faz inclusão.
        # 2. Se a melhor remoção diminui o AIC e a var p-valor > threshold_out no modelo atual, remove.
        # 3. Se nenhuma das ações melhora, paramos.
        
        # Avaliamos inclusão
        do_inclusion = (best_aic_inclusion < best_aic) and (best_pval_inclusion < threshold_in)
        
        # Avaliamos remoção
        # Precisamos verificar qual é o p-valor da var no "modelo atual" para remoção.
        # O p-valor dela no "modelo atual" (best_model) deve estar > threshold_out para justificar remoção
        # E verificar se a remoção melhora AIC também.
        var_pvalues = best_model.pvalues if len(selected_vars) > 0 else {}
        worst_var_pval = 0.0
        if best_candidate_removal is not None and best_candidate_removal in var_pvalues:
            worst_var_pval = var_pvalues[best_candidate_removal]
        do_removal = (
            (best_aic_removal < best_aic) and 
            (worst_var_pval > threshold_out)
        )
        
        # Decidir a ação prioritária
        # Se ambos são possíveis, escolher o que mais reduz o AIC. Se empata, remover ou incluir tanto faz.
        if do_inclusion and do_removal:
            if (best_aic - best_aic_removal) > (best_aic - best_aic_inclusion):
                # Remoção diminui mais o AIC
                do_inclusion = False
            else:
                # Inclusão diminui mais o AIC
                do_removal = False
        
        if do_inclusion:
            # Faz a inclusão
            selected_vars.append(best_candidate_inclusion)
            remaining_vars.remove(best_candidate_inclusion)
            best_model = best_model_inclusion
            best_aic = best_aic_inclusion
            improved = True
            if verbose:
                print(f" + Include {best_candidate_inclusion} with p-value {best_pval_inclusion:.4f}, AIC = {best_aic_inclusion:.2f}")
        
        elif do_removal:
            selected_vars.remove(best_candidate_removal)
            # E recolocamos a var em remaining_vars, se quisermos permitir re-incluir
            # depends on the stepwise design; R "both" permitiria re-incluir
            remaining_vars.append(best_candidate_removal)
            
            best_model = best_model_removal
            best_aic = best_aic_removal
            improved = True
            if verbose:
                print(f" - Remove {best_candidate_removal}, AIC = {best_aic_removal:.2f}")
        
        else:
            # Nenhum movimento melhora o AIC, então paramos
            if verbose:
                print("No further improvement.")
    
    # Construímos o modelo final com as variáveis que ficaram
    return best_model, selected_vars


import statsmodels.formula.api as smf

def backward_selection_pvalue(data, response, alpha=0.05, verbose=True):
    """
    Realiza seleção retrógrada (backward elimination) baseada em p-valores.
    
    Parâmetros
    ----------
    data : pandas.DataFrame
        DataFrame com a coluna de resposta e as colunas preditoras.
    response : str
        Nome da variável resposta (y).
    alpha : float
        Nível de significância para remoção de variáveis (p-valor).
    verbose : bool
        Se True, imprime mensagens a cada remoção de variável.
    
    Retorno
    -------
    final_model : statsmodels RegressionResults
        Modelo final ajustado após o processo de eliminação retrógrada.
    selected_vars : list
        Lista com o nome das variáveis remanescentes no modelo final.
    """
    # Inicia com todas as variáveis, exceto a resposta
    remaining_vars = list(data.columns)
    remaining_vars.remove(response)

    # Ajusta o modelo cheio
    formula = f"{response} ~ {' + '.join(remaining_vars)}"
    model = smf.ols(formula, data=data).fit()

    # Loop até não haver mais p-valores acima de alpha
    while True:
        # pvalores (excluindo o Intercept)
        pvals = model.pvalues.drop('Intercept', errors='ignore')
        
        # Pegamos o p-valor mais alto
        worst_feature = pvals.idxmax()   # nome da variável com maior p-valor
        worst_pval = pvals.max()
        
        if worst_pval > alpha:
            # Remove a variável com maior p-valor
            if verbose:
                print(f"Removendo '{worst_feature}' (p-value = {worst_pval:.4f})")
            remaining_vars.remove(worst_feature)

            # Reajusta o modelo
            if remaining_vars:
                formula = f"{response} ~ {' + '.join(remaining_vars)}"
            else:
                # se não sobrou nenhuma variável, fica só o intercepto
                formula = f"{response} ~ 1"

            model = smf.ols(formula, data=data).fit()
        else:
            # Se não há p-valor acima de alpha, paramos
            break

    # Retorna o modelo final e a lista de variáveis que permaneceram
    return model, remaining_vars
