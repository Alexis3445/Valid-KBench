import os
import pandas as pd
from math import comb
import matplotlib.pyplot as plt

# Parámetros
input_root = "resultados"
output_root = "resultados_oficiales"
os.makedirs(output_root, exist_ok=True)
k = 3

# Función para calcular pass@k oficial
def calcular_passk(c, n, k):
    if c == 0 or k > n:
        return 0.0
    return 1.0 - (comb(n - c, k) / comb(n, k))

# Almacena resultados globales por modelo
resumen_modelos = []

# Procesar carpetas y archivos
for modelo_dir in os.listdir(input_root):
    modelo_path = os.path.join(input_root, modelo_dir)
    if not os.path.isdir(modelo_path):
        continue

    for archivo in os.listdir(modelo_path):
        if not archivo.endswith("-pass@k.csv"):
            continue

        archivo_path = os.path.join(modelo_path, archivo)
        df = pd.read_csv(archivo_path)

        # Agrupación por prompt/lenguaje
        resumen = (
            df.groupby(["modelo", "prompt_id", "categoria", "lenguaje"])
            .agg(aciertos=("acierto", "sum"), total=("acierto", "count"))
            .reset_index()
        )

        # Calcular pass@k oficial
        resumen[f"pass@{k}_oficial"] = resumen.apply(
            lambda row: calcular_passk(row["aciertos"], row["total"], k), axis=1
        )

        # Guardar resultados individuales
        output_file = os.path.join(output_root, f"{modelo_dir}-pass@{k}_oficial.csv")
        resumen.to_csv(output_file, index=False)
        print(f"Guardado: {output_file}")

        # Calcular promedio global para el modelo
        promedio = resumen[f"pass@{k}_oficial"].mean()
        resumen_modelos.append({"modelo": modelo_dir, f"promedio_pass@{k}": promedio})

# Crear y mostrar tabla resumen global
df_resumen = pd.DataFrame(resumen_modelos)
print("\n=== Promedios Globales por Modelo ===")
print(df_resumen.to_string(index=False))

# Guardar CSV con promedios
csv_resumen_path = os.path.join(output_root, f"resumen_global_pass@{k}.csv")
df_resumen.to_csv(csv_resumen_path, index=False)

# Crear y guardar gráfica de barras
plt.figure(figsize=(10, 6))
plt.bar(df_resumen["modelo"], df_resumen[f"promedio_pass@{k}"])
plt.title(f"Promedio Global de pass@{k} por Modelo")
plt.ylabel("pass@k")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
grafica_path = os.path.join(output_root, f"grafica_pass@{k}.png")
plt.savefig(grafica_path)
plt.close()
print(f" Gráfica guardada en: {grafica_path}")
