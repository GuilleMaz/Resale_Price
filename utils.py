import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import norm
import math


# -----------------------
# Calidad de los datos
# -----------------------

def resumen_nulos_ceros(df):
    """
    Genera un Data Frame estilizado con el tipo de dato, cantidad y porcentaje 
    de valores nulos y ceros, compatible con cualquier tipo de dato.
    """

    # Calculamos nulos 
    n_missing = df.isna().sum()

    # Calculamos ceros 
    # Solo intentamos comparar con 0 en columnas numéricas para evitar errores
    # y excluimos booleanos para que False no cuente como 0.
    def contar_ceros(col):
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            return (col == 0).sum()
        else:
            return 0

    n_zeros = df.apply(contar_ceros)

    # Creamos el dataframe descriptivo base
    desc_df = pd.DataFrame({
        'variable': df.columns,
        'variable_type': df.dtypes.astype(str).values,
        'n_missing': n_missing.values,
        'n_zeros': n_zeros.values,
        'complete_rate': 1 - (n_missing.values / len(df))
    })

    # Manipulación mediante Method Chaining
    var_type_missing_df = (
        desc_df
        .assign(
            n_missing_perc = lambda x: (100 * (1 - x['complete_rate'])).round(3),
            n_zeros_perc = lambda x: (100 * (x['n_zeros'] / len(df))).round(3)
        )
        .filter(['variable_type', 'variable', 'n_missing', 'n_missing_perc', 'n_zeros', 'n_zeros_perc'])
        .sort_values(['variable_type', 'n_missing'], ascending=[True, False])
    )

    return (var_type_missing_df.style
            .background_gradient(subset=['n_missing_perc'], cmap='Reds')
            .background_gradient(subset=['n_zeros_perc'], cmap='Blues'))


#-----------------------
# Análisis univariante 
#-----------------------

# Numéricas

def histogram_plot(data, bins=50, cols_per_row=3):
    """
    Genera una cuadrícula de subplots que muestra el histograma y la curva de 
    distribución normal teórica para todas las variables numéricas de un DataFrame.

    Parámetros:
    -----------
    data : pandas.DataFrame
        El conjunto de datos que contiene las variables a graficar.
    bins : int, opcional (por defecto=50)
        El número de barras (intervalos) a utilizar en los histogramas.
    cols_per_row : int, opcional (por defecto=3)
        El número máximo de gráficos a mostrar por cada fila de la cuadrícula.

    Retorna:
    --------
    None
        Muestra la figura en pantalla mediante plt.show().
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    num_vars = len(numeric_cols)
    num_rows = (num_vars + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    axes = axes.flatten() 

    for i, var in enumerate(numeric_cols):
        data_na_omit = data[var].dropna()
        media = data_na_omit.mean()
        desv = data_na_omit.std()
        
        sns.histplot(data_na_omit, bins=bins, stat="density", ax=axes[i], kde=False)

        x = np.linspace(data_na_omit.min(), data_na_omit.max(), 1000)
        axes[i].plot(x, norm.pdf(x, media, desv), color="red", lw=2)
        
        axes[i].set_title(f"Histograma de {var}")
        axes[i].set_ylabel("Densidad")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def plot_all_boxplots(df, cols_per_row=3):
    """
    Identifica variables numéricas y dibuja un boxplot para cada una.
    
    Args:
        df (pd.DataFrame): El dataset a analizar.
        cols_per_row (int): Cuántos gráficos mostrar por fila.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("No se encontraron variables numéricas en el dataset.")
        return

    num_plots = len(numeric_cols)
    rows = math.ceil(num_plots / cols_per_row)
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
    axes = axes.flatten() 

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="skyblue", flierprops={"marker": "o", "markerfacecolor": "red"})
        axes[i].set_title(f'Boxplot de {col}', fontsize=12)
        axes[i].set_ylabel('Valor')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()









# Categóricas


import math
import matplotlib.pyplot as plt
import seaborn as sns

import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categoricas(df, max_categories=20):
    """
    Genera gráficos de barras (frecuencias) optimizados para todas las variables
    categóricas de un DataFrame, limitando el número máximo de categorías a graficar.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El conjunto de datos que contiene las variables categóricas.
    max_categories : int, opcional (por defecto=20)
        El número máximo de categorías principales que se graficarán por variable.
        Previene problemas de rendimiento y legibilidad con variables de alta cardinalidad.

    Retorna:
    --------
    None
        Muestra la figura en pantalla mediante plt.show() o imprime un mensaje
        si no hay variables categóricas.
    """
    cat_cols = df.select_dtypes(include=['category']).columns
    n_vars = len(cat_cols)
   
    if n_vars == 0: return print("No hay variables categóricas.")

    n_cols = 3
    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, col in enumerate(cat_cols):
        # --- OPTIMIZACIÓN 1: Pre-calcular frecuencias con Pandas ---
        # Esto es mucho más rápido que dejar que Seaborn cuente 200k filas
        counts = df[col].value_counts().head(max_categories)
       
        # --- OPTIMIZACIÓN 2: Usar barplot con datos ya resumidos ---
        sns.barplot(x=counts.index, y=counts.values, ax=axes[i], color='skyblue')
       
        axes[i].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
        
        # --- MODIFICACIÓN: Alineación perfecta de las etiquetas ---
        # Fijamos las marcas y luego rotamos anclando el texto por la derecha
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(counts.index, rotation=45, ha='right')
       
        # --- OPTIMIZACIÓN 3: Anotaciones eficientes ---
        # Solo ponemos etiquetas si no hay una cantidad absurda de barras
        if len(counts) <= max_categories:
            for p in axes[i].patches:
                axes[i].annotate(f'{int(p.get_height())}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                                textcoords='offset points')

    # Limpiar ejes sobrantes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




#---------------------
# Análisis Bivariante
#---------------------

import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots_vs_target(data, target_var, show_outliers=True, cols_per_row=3, max_categories=20):
    """
    Genera boxplots de las variables categóricas frente a una variable target,
    limitando el número máximo de categorías a graficar.
    
    Parámetros:
    - data: DataFrame de pandas.
    - target_var: Nombre de la columna numérica objetivo (ej. 'PRICE').
    - show_outliers: Booleano. Si es False, oculta los puntos fuera de los bigotes.
    - cols_per_row: Número de gráficos por fila.
    - max_categories: Número máximo de categorías principales a graficar por variable.
    """
    # 1. Identificar variables categóricas
    cat_cols = data.select_dtypes(include=['category']).columns.tolist()
    
    if not cat_cols:
        print("No se encontraron variables categóricas.")
        return

    # 2. Configurar la estructura de la cuadrícula
    n_vars = len(cat_cols)
    n_rows = math.ceil(n_vars / cols_per_row)
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 6, n_rows * 5))
    axes = axes.flatten() if n_vars > 1 else [axes]

    # 3. Iterar y graficar
    for i, col in enumerate(cat_cols):
        # --- MODIFICACIÓN 1: Limitar el número máximo de categorías ---
        # Ordenamos por la mediana y nos quedamos solo con el top indicado por max_categories
        my_order = data.groupby(col, observed=False)[target_var].median().sort_values(ascending=False).index[:max_categories]
        
        sns.boxplot(
            data=data, 
            x=col, 
            y=target_var, 
            ax=axes[i], 
            order=my_order, 
            palette="viridis",
            hue=col,        
            legend=False,   
            showfliers=show_outliers
        )
        
        # Estética
        axes[i].set_title(f'{target_var} por {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(target_var)
        
        # --- MODIFICACIÓN 2: Alineación perfecta de las etiquetas ---
        # Si hay más de 3 barras o el texto es largo, rotamos y anclamos a la derecha
        if len(my_order) > 3 or data[col].astype(str).str.len().max() > 10:
            axes[i].set_xticks(range(len(my_order)))
            axes[i].set_xticklabels(my_order, rotation=45, ha='right')

    # 4. Eliminar ejes sobrantes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def plot_all_histograms_vs_target(data, target_var, bins=50, cols_per_row=3):
    """
    Genera una cuadrícula de histogramas para todas las variables numéricas 
    de un DataFrame, subdivididas (coloreadas) según una variable objetivo (target).

    Parámetros:
    -----------
    data : pandas.DataFrame
        El conjunto de datos que contiene las variables.
    target_var : str
        El nombre de la variable categórica objetivo que dividirá la distribución (hue).
    bins : int, opcional (por defecto=50)
        El número de barras (intervalos) a utilizar en los histogramas.
    cols_per_row : int, opcional (por defecto=3)
        El número máximo de gráficos a mostrar por cada fila de la cuadrícula.

    Excepciones:
    ------------
    ValueError
        Si `target_var` no se encuentra en las columnas del DataFrame.

    Retorna:
    --------
    None
        Muestra la figura en pantalla mediante plt.show().
    """
    if target_var not in data.columns:
        raise ValueError(f"La variable objetivo '{target_var}' no existe en el dataframe")

    # 1. Identificar variables numéricas (excluyendo la variable target si fuera numérica)
    numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != target_var]
    num_vars = len(numeric_cols)
    
    if num_vars == 0:
        print("No se encontraron variables numéricas para graficar.")
        return

    # 2. Configurar la estructura de la cuadrícula
    num_rows = math.ceil(num_vars / cols_per_row)
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    
    # Aplanar el array de ejes para iterar fácilmente
    axes = axes.flatten() if num_vars > 1 else [axes]

    # 3. Iterar sobre las variables numéricas y crear los gráficos
    for i, var in enumerate(numeric_cols):
        # Filtrar NAs solo para la variable en curso y el target
        data_na_omit = data[[var, target_var]].dropna(subset=[var])
        
        sns.histplot(
            data=data_na_omit,
            x=var,
            hue=target_var,
            bins=bins,
            stat="density",
            common_norm=False, # Permite que las densidades de ambos grupos se vean bien aunque tengan distinto tamaño
            alpha=0.5,
            ax=axes[i]
        )
        
        # Estética del subgráfico
        axes[i].set_title(f"{var} según {target_var}", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Densidad")
        axes[i].set_xlabel('')

    # 4. Eliminar ejes sobrantes (si n_vars no es múltiplo de cols_per_row)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()





def boxplot_var_target_plot(data, target_var, cols_per_row=3):
    """
    Genera una cuadrícula de diagramas de caja (boxplots) para todas las 
    variables numéricas de un DataFrame, agrupadas según una variable 
    categórica objetivo (target).

    Parámetros:
    -----------
    data : pandas.DataFrame
        El conjunto de datos que contiene las variables a analizar.
    target_var : str
        El nombre de la variable categórica objetivo que dividirá los datos (eje X).
    cols_per_row : int, opcional (por defecto=3)
        El número de gráficos que se mostrarán por cada fila en la cuadrícula.

    Retorna:
    --------
    None
        Muestra la figura en pantalla mediante plt.show() o imprime un mensaje 
        si no se encuentran variables numéricas.
    """
    # 1. Filtramos: solo columnas numéricas y que NO sean la variable objetivo
    vars_to_plot = [col for col in data.select_dtypes(include=[np.number]).columns if col != target_var]
    
    num_vars = len(vars_to_plot)
    if num_vars == 0:
        print("No se encontraron variables numéricas para graficar.")
        return

    # 2. Configuramos la cuadrícula (filas x columnas)
    num_rows = (num_vars + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    
    # Aplanamos los ejes para iterar sobre ellos fácilmente (en caso de que sea una matriz)
    if num_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes] # Si solo hay una variable, lo volvemos lista para que el loop no falle

    # 3. Generamos los gráficos
    for i, var in enumerate(vars_to_plot):
        # Omitimos NAs solo de la variable actual y el target
        data_na_omit = data[[var, target_var]].dropna(subset=[var])
        
        sns.boxplot(data=data_na_omit, x=target_var, y=var, ax=axes[i])
        
        # Estética personalizada
        axes[i].set_title(f"{var} según {target_var}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel(target_var)
        axes[i].set_ylabel(var)

    # 4. Limpiamos los espacios vacíos si el número de gráficas es impar
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




#------------------------
# Análisis multivariante
#------------------------

def corrplot(data, show_annot=True):
    """
    Calcula la matriz de correlación de Pearson y la visualiza en un heatmap.

    Parámetros:
    -----------
    data : pandas.DataFrame
        El conjunto de datos con las variables numéricas a correlacionar.
    show_annot : bool, opcional (por defecto=True)
        Si es True, muestra los valores numéricos dentro de cada celda.
        Si es False, solo muestra los colores.
    """
    # Filtramos solo numéricas para evitar advertencias de pandas
    correlacion_pearson_df = data.select_dtypes(include=[np.number]).corr(method='pearson')

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    
    # Usamos el parámetro show_annot y añadimos fmt para que los números no se amontonen
    sns.heatmap(correlacion_pearson_df, 
                annot=show_annot, 
                fmt='.2f' if show_annot else '', 
                cmap='coolwarm')
    
    plt.title('Matriz de Correlación de Pearson')
    plt.show()







