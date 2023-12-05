import random

def seleccionar_con_reemplazo(lista, n):
    """
    Selecciona n elementos de la lista con reemplazo.

    Parameters:
    - lista: Lista de cadenas.
    - n: Número de elementos a seleccionar.

    Returns:
    - Lista de n elementos seleccionados con reemplazo.
    """
    seleccionados = random.choices(lista, k=n)
    return seleccionados

# Ejemplo de uso
lista_de_cadenas = ["A", "B", "C", "D", "E"]
numero_de_selecciones = 3

resultados = seleccionar_con_reemplazo(lista_de_cadenas, numero_de_selecciones)
print("Cadenas seleccionadas:", resultados)
