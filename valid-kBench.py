import os
import json
import time
import re
import subprocess
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from math import comb

class ChatModelHF:
    def __init__(self, model_name, device="cuda"):
        print(f"[INFO] Cargando modelo {model_name} en {device}")
        if "qwen" in model_name.lower():
            print("[INFO] Desactivando SDPA para compatibilidad con Qwen")
            os.environ["PYTORCH_USE_SDPA"] = "0"

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.device = device
        self.chat_history = []

    def ask(self, prompt, reset_history=True, k=1):
        if reset_history:
            self.chat_history = []

        full_prompt = "\n".join(self.chat_history + [prompt])
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            t1 = time.perf_counter()

        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True)[len(full_prompt):].strip()
            for output in outputs
        ]

        return responses, (t1 - t0) / k

    @staticmethod
    def extract_code(text, language):
        jupyter_blocks = re.findall(r"<jupyter_code>(.*?)</jupyter_code>", text, re.DOTALL)
        if jupyter_blocks:
            return jupyter_blocks[0].strip()

        text = re.split(r"<jupyter_output>|<jupyter_text>|<jupyter_code>", text)[0]

        pattern_md = rf"```{language}(.*?)```"
        match = re.search(pattern_md, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = text.strip().splitlines()
        code_lines = []
        inside_code = False
        for line in lines:
            stripped = line.strip()
            if language == "python" and stripped.startswith(("def ", "print(", "for ", "if ", "while ", "#", "import ", "return", "class ")):
                inside_code = True
            elif language == "bash" and stripped.startswith(("echo ", "#!", "for ", "if ", "while ")):
                inside_code = True
            elif language == "c" and stripped.startswith(("#include", "int main", "printf", "scanf")):
                inside_code = True

            if inside_code:
                if "<jupyter_" in stripped:
                    break
                code_lines.append(line)

        return "\n".join(code_lines).strip() if code_lines else ""

    @staticmethod
    def cargar_prompts(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class CodeValidator:
    def validar(self, code, language, entrada):
        ext = {"python": ".py", "c": ".c", "bash": ".bash"}
        archivo = f"codigo_gen{ext[language]}"
        with open(archivo, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            t0 = time.perf_counter()
            if language == "python":
                proc = subprocess.run(["python", archivo], input=str(entrada), capture_output=True, text=True)
            elif language == "c":
                subprocess.run(["gcc", archivo, "-o", "prog"], check=True)
                proc = subprocess.run(["./prog"], input=str(entrada), capture_output=True, text=True)
            elif language == "bash":
                proc = subprocess.run(["bash", archivo], input=str(entrada), capture_output=True, text=True)
            else:
                return False, "Lenguaje no soportado", None
            t1 = time.perf_counter()
            return (proc.returncode == 0, proc.stdout.strip(), t1 - t0)
        except Exception as e:
            return False, str(e), None


class Evaluador:
    def __init__(self, model_alias, model_id):
        self.model_alias = model_alias
        self.model = ChatModelHF(model_id)
        self.validator = CodeValidator()
        self.resultados = []
        self.auditoria_dir = f"auditoria/{model_alias}"
        os.makedirs(self.auditoria_dir, exist_ok=True)
        self.audit_file = os.path.join(self.auditoria_dir, "auditoria.txt")

    def log_auditoria(self, prompt, response, code, lang, esperado, salida):
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Lenguaje: {lang}\n")
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"Respuesta del modelo:\n{response}\n\n")
            f.write(f"Código extraído:\n{code if code else '[NO DETECTADO]'}\n\n")
            f.write(f"Esperado: {esperado}\n")
            f.write(f"Salida   : {salida}\n")

    def evaluar(self, prompts_json):
        prompts = ChatModelHF.cargar_prompts(prompts_json)
        k = 3
        id_counter = 0
        for categoria, tests in tqdm(prompts.items(), desc=" Categorías", position=0):
            for test in tqdm(tests, desc=f" Prompts ({categoria})", leave=False, position=1):
                desc = test["description"]
                entrada = test["input"]
                esperado = str(test["expected_output"]).strip()
                prompt_id = f"{categoria}_{id_counter}"
                id_counter += 1
                for lang in ["python", "c", "bash"]:
                    prompt = f"""Write a program in the language {lang} that {desc}: {entrada}.
The program must print **only** the final result (no extra text, no explanations, no labels), and provide the following expected output: {esperado}.
Finish your code right after the final output line. Do not add any additional text, comments, or tasks.""".strip()

                    respuestas, t_modelo = self.model.ask(prompt, reset_history=True, k=k)
                    aciertos = 0
                    for i, respuesta in enumerate(respuestas):
                        codigo = self.model.extract_code(respuesta, lang)
                        if not codigo:
                            self.resultados.append({
                                "modelo": self.model_alias,
                                "prompt_id": prompt_id,
                                "categoria": categoria,
                                "lenguaje": lang,
                                "intento": i + 1,
                                "tiempo_modelo": round(t_modelo, 4),
                                "tiempo_ejecucion": 0.0,
                                "acierto": 0
                            })
                            continue
                        correcto, salida, t_ejec = self.validator.validar(codigo, lang, entrada)
                        acierto = int(correcto and salida == esperado)
                        self.log_auditoria(f"[Try {i+1}] " + prompt, respuesta, codigo, lang, esperado, salida)
                        self.resultados.append({
                            "modelo": self.model_alias,
                            "prompt_id": prompt_id,
                            "categoria": categoria,
                            "lenguaje": lang,
                            "intento": i + 1,
                            "tiempo_modelo": round(t_modelo, 4),
                            "tiempo_ejecucion": round(t_ejec or 0, 4),
                            "acierto": acierto
                        })

    def exportar_resultados(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.resultados)
        passk_por_prompt = df.groupby(["prompt_id", "categoria", "lenguaje"]).agg({
            "acierto": lambda x: 1.0 if x.sum() > 0 else 0.0
        }).rename(columns={"acierto": "pass@k"}).reset_index()
        df_final = df.merge(passk_por_prompt, on=["prompt_id", "categoria", "lenguaje"], how="left")
        df_final.to_csv(path, index=False)
        print(f" Resultados guardados en {path}")

    def graficar(self, out_dir, k=3):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.resultados)
        if df.empty:
            print(" No hay resultados para graficar.")
            return

        modelos = df["modelo"].unique()
        for modelo in modelos:
            df_modelo = df[df["modelo"] == modelo]
            df_agg = df_modelo.groupby("lenguaje").agg({
                "tiempo_modelo": "mean",
                "tiempo_ejecucion": "mean",
                "acierto": ["sum", "count"]
            })
            df_agg.columns = ['t_modelo', 't_ejecucion', 'aciertos', 'intentos']
            df_agg["porcentaje"] = df_agg["aciertos"] / df_agg["intentos"] * 100

            fig, ax = plt.subplots(figsize=(8, 6))
            df_agg["porcentaje"].plot(kind="bar", color="skyblue", ax=ax)
            ax.set_title(f"Aciertos (pass@{k}) por lenguaje - {modelo}")
            ax.set_ylabel("Porcentaje")
            ax.set_ylim(0, 100)
            plt.xticks(rotation=0)
            plt.tight_layout()
            path = os.path.join(out_dir, f"{modelo}_aciertos.png")
            plt.savefig(path)
            plt.close()
            print(f" Gráfico guardado en {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluador de modelos de código con Hugging Face")
    parser.add_argument("--alias", required=True, help="Nombre corto del modelo (para carpetas)")
    parser.add_argument("--model_id", required=True, help="ID del modelo en Hugging Face")
    parser.add_argument("--prompts", default="prompts.json", help="Ruta al archivo JSON de prompts")
    args = parser.parse_args()

    evaluador = Evaluador(args.alias, args.model_id)
    evaluador.evaluar(args.prompts)

    out_dir_resultados = f"resultados/{args.alias}"
    out_file = f"{out_dir_resultados}/{args.alias}-pass@k.csv"
    evaluador.exportar_resultados(path=out_file)

    out_dir_graficas = f"graficas/{args.alias}"
    evaluador.graficar(out_dir=out_dir_graficas, k=3)


if __name__ == "__main__":
    main()
