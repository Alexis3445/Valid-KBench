import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Paleta pastel suave
PALETA_TENUE = [
    "#a6cee3", "#b2df8a", "#fb9a99",
    "#fdbf6f", "#cab2d6", "#ffff99",
    "#1f78b4", "#33a02c"
]

def cargar_resultados(carpeta_resultados):
    modelos_data = {}
    for nombre_modelo in os.listdir(carpeta_resultados):
        ruta_modelo = os.path.join(carpeta_resultados, nombre_modelo)
        if os.path.isdir(ruta_modelo):
            for archivo in os.listdir(ruta_modelo):
                if archivo.endswith(".csv") and nombre_modelo in archivo:
                    ruta_csv = os.path.join(ruta_modelo, archivo)
                    try:
                        df = pd.read_csv(ruta_csv)
                        modelos_data[nombre_modelo] = df
                        print(f"Loaded: {ruta_csv}")
                    except Exception as e:
                        print(f"Error loading {ruta_csv}: {e}")
                    break
            else:
                print(f"No CSV file found for {nombre_modelo}")
    return modelos_data

def agregar_etiquetas(ax, bars, decimal_places=1, fontsize=8, rotation=45, offset_y=6):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.{decimal_places}f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=fontsize,
            rotation=rotation
        )

def graficar_correctitud(modelos_data, carpeta_out):
    lenguajes = set()
    data = {}

    for modelo, df in modelos_data.items():
        df_agg = df.groupby("lenguaje")["acierto"].agg(["sum", "count"])
        df_agg["accuracy (%)"] = df_agg["sum"] / df_agg["count"] * 100
        data[modelo] = df_agg["accuracy (%)"]
        lenguajes.update(df_agg.index.tolist())

    lenguajes = sorted(lenguajes)
    x = np.arange(len(lenguajes))
    ancho = 0.8 / len(data) if data else 0.2

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (modelo, porcentajes) in enumerate(data.items()):
        valores = [porcentajes.get(lang, 0) for lang in lenguajes]
        bars = ax.bar(x + i * ancho, valores, width=ancho, label=modelo, color=PALETA_TENUE[i % len(PALETA_TENUE)])
        agregar_etiquetas(ax, bars, decimal_places=1, fontsize=6, rotation=90, offset_y=4)

    ax.set_xticks(x + ancho * (len(data) - 1) / 2)
    ax.set_xticklabels(lenguajes)
    ax.set_ylabel("Correctness (%)")
    ax.set_title("Code Correctness by Language")
    ax.set_ylim(0, 110)  # Espacio extra para etiquetas
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    out_path = os.path.join(carpeta_out, "code_correctness.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Correctness graph saved to: {out_path}")

def graficar_tiempos(modelos_data, carpeta_out, tipo):
    assert tipo in ["tiempo_modelo", "tiempo_ejecucion"]
    lenguajes = set()
    data = {}

    for modelo, df in modelos_data.items():
        df_agg = df.groupby("lenguaje")[tipo].mean()
        data[modelo] = df_agg
        lenguajes.update(df_agg.index.tolist())

    lenguajes = sorted(lenguajes)
    x = np.arange(len(lenguajes))
    ancho = 0.8 / len(data) if data else 0.2

    # Calcular el valor máximo para ajustar ylim
    max_val = max([valores.get(lang, 0) for lang in lenguajes for valores in data.values()])

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (modelo, valores) in enumerate(data.items()):
        datos = [valores.get(lang, 0) for lang in lenguajes]
        bars = ax.bar(x + i * ancho, datos, width=ancho, label=modelo, color=PALETA_TENUE[i % len(PALETA_TENUE)])

        agregar_etiquetas(ax, bars, decimal_places=3, fontsize=6, rotation=90, offset_y=4)

    ax.set_xticks(x + ancho * (len(data) - 1) / 2)
    ax.set_xticklabels(lenguajes)
    ax.set_ylabel("Inference Time (s)" if tipo == "tiempo_modelo" else "Execution Time (s)")
    ax.set_title("Model Inference Time by Language" if tipo == "tiempo_modelo" else "Program Execution Time by Language")
    ax.set_ylim(0, max_val * 1.15)  # Espacio superior para etiquetas
    ax.legend(title="Model", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    nombre = "inference_time.png" if tipo == "tiempo_modelo" else "execution_time.png"
    out_path = os.path.join(carpeta_out, nombre)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Timing graph saved to: {out_path}")

def main():
    carpeta_resultados = "resultados"
    carpeta_salida = "graficas_comparativas"
    os.makedirs(carpeta_salida, exist_ok=True)

    modelos_data = cargar_resultados(carpeta_resultados)

    if not modelos_data:
        print("No data found to plot.")
        return

    graficar_correctitud(modelos_data, carpeta_salida)
    graficar_tiempos(modelos_data, carpeta_salida, tipo="tiempo_modelo")
    graficar_tiempos(modelos_data, carpeta_salida, tipo="tiempo_ejecucion")

if __name__ == "__main__":
    main()
