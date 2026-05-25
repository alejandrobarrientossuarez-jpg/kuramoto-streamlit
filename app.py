# ============================================================
# Simulador de Sincronización Discreta tipo Kuramoto sobre Grafos
# Adaptado para Streamlit
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import streamlit as st
from itertools import product

st.set_page_config(
    page_title="Simulador Kuramoto Discreto",
    page_icon="🔄",
    layout="wide",
)

# ============================================================
# 1. Funciones auxiliares
# ============================================================

def bell_number(n):
    bell = [[0] * (n + 1) for _ in range(n + 1)]
    bell[0][0] = 1
    for i in range(1, n + 1):
        bell[i][0] = bell[i - 1][i - 1]
        for j in range(1, i + 1):
            bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]
    return bell[n][0]


def kuramoto_order_parameter(theta, M):
    z = np.exp(2j * np.pi * np.array(theta, dtype=int) / M)
    return float(np.abs(np.mean(z)))


# ============================================================
# 2. Clase principal del modelo
# ============================================================

class KuramotoDiscreto:
    def __init__(self, N=4, M=6, kappa=2, omega=0, tipo_grafo="Completo",
                 epsilon=1, epsilon_modo="estricto"):
        self.N = int(N)
        self.M = int(M)
        self.kappa = int(kappa)
        self.omega = int(omega)
        self.tipo_grafo = tipo_grafo
        self.epsilon = float(epsilon)
        self.epsilon_modo = epsilon_modo
        self.G = self._crear_grafo()
        self.pos = nx.spring_layout(self.G, seed=7)

    def _crear_grafo(self):
        tipos = {
            "Completo": nx.complete_graph(self.N),
            "Camino":   nx.path_graph(self.N),
            "Ciclo":    nx.cycle_graph(self.N),
            "Estrella": nx.star_graph(self.N - 1),
        }
        return tipos[self.tipo_grafo]

    def dM_signed(self, a, b):
        return ((int(a) - int(b) + self.M // 2) % self.M) - self.M // 2

    def dM_circular(self, a, b):
        diff = abs(int(a) - int(b))
        return min(diff, self.M - diff)

    def sigma_kappa(self, S):
        if S >= self.kappa:
            return 1
        elif S <= -self.kappa:
            return -1
        else:
            return 0

    def S_i(self, theta, i):
        theta = np.array(theta, dtype=int)
        return sum(self.dM_signed(theta[j], theta[i]) for j in self.G.neighbors(i))

    def vector_S(self, theta):
        return np.array([self.S_i(theta, i) for i in range(self.N)], dtype=int)

    def vector_sigma(self, theta):
        return np.array([self.sigma_kappa(s) for s in self.vector_S(theta)], dtype=int)

    def paso(self, theta):
        theta = np.array(theta, dtype=int)
        return (theta + self.omega + self.vector_sigma(theta)) % self.M

    def simular(self, theta0, T=30):
        theta = np.array(theta0, dtype=int) % self.M
        trayectoria = [theta.copy()]
        S_hist, sigma_hist = [], []
        r_hist = [kuramoto_order_parameter(theta, self.M)]
        subredes = [self._subred_edges(theta)]

        for _ in range(T):
            S_t = self.vector_S(theta)
            sig_t = self.vector_sigma(theta)
            S_hist.append(S_t.copy())
            sigma_hist.append(sig_t.copy())
            theta = self.paso(theta)
            trayectoria.append(theta.copy())
            r_hist.append(kuramoto_order_parameter(theta, self.M))
            subredes.append(self._subred_edges(theta))

        return {
            "trayectoria": np.array(trayectoria),
            "S_hist":      np.array(S_hist),
            "sigma_hist":  np.array(sigma_hist),
            "r_hist":      np.array(r_hist),
            "subredes":    subredes,
        }

    def _subred_edges(self, theta):
        theta = np.array(theta, dtype=int)
        estricto = (self.epsilon_modo == "estricto")
        edges = []
        for i, j in self.G.edges():
            d = self.dM_circular(theta[i], theta[j])
            if (d < self.epsilon) if estricto else (d <= self.epsilon):
                edges.append((i, j))
        return tuple(sorted(edges))

    def subred_acuerdo_edges(self, theta):
        return self._subred_edges(theta)

    def esta_sincronizado(self, theta):
        theta = np.array(theta, dtype=int)
        return bool(np.all(theta == theta[0]))

    def clasificar_atractor(self, theta0, T_max=200):
        theta = np.array(theta0, dtype=int) % self.M
        vistos = {}
        for t in range(T_max + 1):
            key = tuple(theta.tolist())
            if self.esta_sincronizado(theta):
                theta_next = self.paso(theta)
                if self.esta_sincronizado(theta_next) and np.all(theta_next == theta):
                    periodo = 1
                else:
                    periodo = 1
                    tmp = theta_next.copy()
                    for p in range(1, self.M + 1):
                        if np.all(tmp == theta):
                            periodo = p
                            break
                        tmp = self.paso(tmp)
                return {"tipo": "sync", "tiempo": t, "periodo": periodo,
                        "estado_final": theta.copy()}
            if key in vistos:
                t_inicio = vistos[key]
                periodo = t - t_inicio
                tipo = "punto fijo no trivial" if periodo == 1 else "ciclo"
                return {"tipo": tipo, "tiempo": t_inicio, "periodo": periodo,
                        "estado_final": theta.copy()}
            vistos[key] = t
            theta = self.paso(theta)
        return {"tipo": "transitorio", "tiempo": T_max, "periodo": None,
                "estado_final": theta.copy()}

    def tabla_trayectoria(self, resultado):
        trayectoria = resultado["trayectoria"]
        S_hist      = resultado["S_hist"]
        sigma_hist  = resultado["sigma_hist"]
        r_hist      = resultado["r_hist"]
        subredes    = resultado["subredes"]
        T           = len(trayectoria) - 1
        filas = []
        for t in range(T + 1):
            fila = {
                "t":              t,
                "theta(t)": tuple(int(x) for x in trayectoria[t]),
                "r(t)":           round(r_hist[t], 4),
                "subred_acuerdo": tuple((int(a), int(b)) for a, b in resultado["subredes"][t]),
            }
            if t < T:
                fila["S_i(theta_t)"] = tuple(int(x) for x in S_hist[t])
                fila["sigma_kappa(S_i)"] = tuple(int(x) for x in sigma_hist[t])
            else:
                fila["S_i(theta_t)"]     = "-"
                fila["sigma_kappa(S_i)"] = "-"
            filas.append(fila)
        return pd.DataFrame(filas)


# ============================================================
# 3. Exploración exhaustiva
# ============================================================

def exploracion_exhaustiva(N, M, kappa, omega=0, tipo_grafo="Completo",
                            epsilon=1, epsilon_modo="estricto", T_max=150):
    modelo = KuramotoDiscreto(N=N, M=M, kappa=kappa, omega=omega,
                               tipo_grafo=tipo_grafo, epsilon=epsilon,
                               epsilon_modo=epsilon_modo)
    total = M ** N
    conteo = {"sync": 0, "punto fijo no trivial": 0, "ciclo": 0, "transitorio": 0}
    tiempos_sync = []
    subredes_realizables = set()
    transiciones = set()

    for theta0 in product(range(M), repeat=N):
        resultado = modelo.simular(theta0, T=min(T_max, 50))
        clasif    = modelo.clasificar_atractor(theta0, T_max=T_max)
        conteo[clasif["tipo"]] += 1
        if clasif["tipo"] == "sync":
            tiempos_sync.append(clasif["tiempo"])
        subredes = resultado["subredes"]
        for s in subredes:
            subredes_realizables.add(s)
        for a, b in zip(subredes[:-1], subredes[1:]):
            if a != b:
                transiciones.add((a, b))

    return {
        "total_configuraciones":   total,
        "sync":                    conteo["sync"],
        "fix":                     conteo["punto fijo no trivial"],
        "cyc":                     conteo["ciclo"],
        "transitorio":             conteo["transitorio"],
        "rho_sync":                conteo["sync"] / total,
        "rho_fix":                 conteo["punto fijo no trivial"] / total,
        "rho_cyc":                 conteo["ciclo"] / total,
        "rho_transitorio":         conteo["transitorio"] / total,
        "tiempos_sync":            tiempos_sync,
        "num_subredes_realizables": len(subredes_realizables),
        "num_transiciones":        len(transiciones),
        "Bell_BN":                 bell_number(N),
        "subredes_realizables":    subredes_realizables,
        "transiciones":            transiciones,
    }


# ============================================================
# 4. Barrido de kappa
# ============================================================

def barrido_kappa(N, M, omega=0, tipo_grafo="Completo",
                  epsilon=1, epsilon_modo="estricto", T_max=150):
    modelo_ref = KuramotoDiscreto(N=N, M=M, kappa=1, tipo_grafo=tipo_grafo)
    degmax     = max(dict(modelo_ref.G.degree()).values())
    kappa_max  = degmax * (M // 2)
    filas = []
    for kappa in range(1, kappa_max + 1):
        res = exploracion_exhaustiva(N=N, M=M, kappa=kappa, omega=omega,
                                     tipo_grafo=tipo_grafo, epsilon=epsilon,
                                     epsilon_modo=epsilon_modo, T_max=T_max)
        filas.append({
            "kappa":                kappa,
            "rho_sync":             res["rho_sync"],
            "rho_fix":              res["rho_fix"],
            "rho_cyc":              res["rho_cyc"],
            "rho_transitorio":      res["rho_transitorio"],
            "subredes_realizables": res["num_subredes_realizables"],
            "transiciones":         res["num_transiciones"],
            "Bell_BN":              res["Bell_BN"],
        })
    return pd.DataFrame(filas)


# ============================================================
# 5. Visualizaciones
# ============================================================

def fig_heatmap(resultado):
    trayectoria = resultado["trayectoria"]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(trayectoria.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Tiempo t")
    ax.set_ylabel("Vértice i")
    ax.set_title("Heatmap de evolución de fases")
    plt.colorbar(im, ax=ax, label="Fase en Z_M")
    plt.tight_layout()
    return fig


def fig_order_parameter(resultado):
    r = resultado["r_hist"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(r)), r, marker="o")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Tiempo t")
    ax.set_ylabel("r(t)")
    ax.set_title("Parámetro de orden de Kuramoto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def fig_grafo(modelo, theta, title="Grafo coloreado por fase"):
    theta = np.array(theta, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx_edges(modelo.G, modelo.pos, ax=ax, alpha=0.25)
    acuerdo = modelo.subred_acuerdo_edges(theta)
    if acuerdo:
        nx.draw_networkx_edges(modelo.G, modelo.pos, edgelist=acuerdo, ax=ax, width=3)
    nodes = nx.draw_networkx_nodes(
        modelo.G, modelo.pos,
        node_color=theta.astype(float),
        cmap=plt.cm.viridis,
        vmin=0, vmax=modelo.M - 1,
        node_size=650, ax=ax,
    )
    nx.draw_networkx_labels(modelo.G, modelo.pos, ax=ax)
    plt.colorbar(nodes, ax=ax, label="Fase")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_hist_tiempos(tiempos):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(tiempos, bins=range(min(tiempos), max(tiempos) + 2), align="left", rwidth=0.85)
    ax.set_xlabel("Tiempo de sincronización")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de tiempos de sincronización")
    plt.tight_layout()
    return fig


def fig_mapa_regimenes(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["kappa"], df["rho_sync"], marker="o", label="rho_sync")
    ax.plot(df["kappa"], df["rho_fix"],  marker="o", label="rho_fix")
    ax.plot(df["kappa"], df["rho_cyc"],  marker="o", label="rho_cyc")
    ax.set_xlabel("kappa")
    ax.set_ylabel("Proporción")
    ax.set_title("Mapa de regímenes por barrido de kappa")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def fig_diagrama_transiciones(transiciones, max_labels=20):
    D = nx.DiGraph()
    for a, b in transiciones:
        D.add_node(a)
        D.add_node(b)
        D.add_edge(a, b)
    if D.number_of_nodes() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(D, seed=11)
    nx.draw_networkx_nodes(D, pos, node_size=500, ax=ax)
    nx.draw_networkx_edges(D, pos, arrows=True, arrowstyle="->", arrowsize=15, ax=ax)
    if D.number_of_nodes() <= max_labels:
        nx.draw_networkx_labels(D, pos, labels={n: str(n) for n in D.nodes()},
                                font_size=7, ax=ax)
    ax.set_title("Diagrama de transiciones de subredes")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================
# 6. Conclusiones (devuelven strings para Streamlit)
# ============================================================

def conclusion_simulacion(modelo, resultado, clasif, T):
    tipo    = clasif["tipo"]
    r_final = float(resultado["r_hist"][-1])
    r_ini   = float(resultado["r_hist"][0])
    theta_final = tuple(clasif["estado_final"])
    lines = []

    if tipo == "sync":
        t_sync  = clasif["tiempo"]
        periodo = clasif["periodo"]
        fase    = theta_final[0]
        if periodo == 1:
            lines.append("✅ **LA SUBRED SINCRONIZÓ** (punto fijo global)")
            lines.append(f"Todos los vértices alcanzaron la fase **{fase}** en t = {t_sync}.")
            lines.append(f"El sistema permanece en θ = {theta_final} para todo t ≥ {t_sync}.")
        else:
            lines.append("✅ **LA SUBRED SINCRONIZÓ** (rotación global)")
            lines.append("Todos los vértices coinciden en fase y rotan juntos.")
            lines.append(f"Periodo del ciclo sincronizado: **{periodo}** pasos. Alcanzado en t = {t_sync}.")
        lines.append(f"Parámetro de orden final: **r(T) = {r_final:.4f}**")

    elif tipo == "punto fijo no trivial":
        lines.append("❌ **LA SUBRED NO SINCRONIZÓ** (punto fijo no trivial)")
        lines.append(f"El sistema convergió a un estado fijo con fases distintas: θ = {theta_final}")
        lines.append(f"r(0) = {r_ini:.4f}  →  r(T) = {r_final:.4f}")

    elif tipo == "ciclo":
        periodo  = clasif["periodo"]
        t_inicio = clasif["tiempo"]
        lines.append(f"❌ **LA SUBRED NO SINCRONIZÓ** (ciclo de periodo {periodo})")
        lines.append(f"Entró en un ciclo de periodo {periodo} a partir de t = {t_inicio}.")
        lines.append(f"Estado en t = {T}: θ = {theta_final}")
        lines.append(f"r(0) = {r_ini:.4f}  →  r(T) = {r_final:.4f}")

    else:
        lines.append(f"❓ **RESULTADO INDETERMINADO** (transitorio en T = {T})")
        lines.append(f"No se detectó sincronización ni ciclo en {T} pasos.")
        lines.append(f"Estado en t = {T}: θ = {theta_final}")
        lines.append(f"r(0) = {r_ini:.4f}  →  r(T) = {r_final:.4f}")

    return "\n\n".join(lines)


def conclusion_exhaustiva(res, N, M, kappa, tipo_grafo):
    rho_sync = res["rho_sync"]
    rho_fix  = res["rho_fix"]
    rho_cyc  = res["rho_cyc"]
    total    = res["total_configuraciones"]
    regimenes = {
        "sincronización":        rho_sync,
        "punto fijo no trivial": rho_fix,
        "ciclo":                 rho_cyc,
        "transitorio":           res["rho_transitorio"],
    }
    dominante = max(regimenes, key=regimenes.get)
    lines = [f"**Grafo:** {tipo_grafo} | N={N} | M={M} | κ={kappa}",
             f"**Espacio total:** {total} configuraciones", ""]

    if rho_sync >= 0.5:
        lines.append(f"✅ La **MAYORÍA** de las condiciones iniciales sincronizan ({rho_sync*100:.1f}%).")
    elif rho_sync > 0:
        lines.append(f"⚠️ Solo una **minoría** sincroniza ({rho_sync*100:.1f}%). Régimen dominante: {dominante}.")
    else:
        lines.append(f"❌ **NINGUNA** condición inicial sincronizó. Régimen dominante: {dominante}.")

    if res["tiempos_sync"]:
        t_med = float(np.mean(res["tiempos_sync"]))
        t_max = max(res["tiempos_sync"])
        lines.append(f"Tiempo medio de sync: **{t_med:.1f}** pasos | máximo: **{t_max}** pasos.")

    return "\n\n".join(lines)


def conclusion_barrido(df):
    k_opt   = int(df.loc[df["rho_sync"].idxmax(), "kappa"])
    rho_max = float(df["rho_sync"].max())
    k_min   = int(df["kappa"].min())
    k_max_df = int(df["kappa"].max())
    lines = [f"Rango de κ explorado: {k_min} … {k_max_df}", ""]

    if rho_max == 0.0:
        lines.append("❌ No se observó sincronización en ningún valor de κ.")
    else:
        lines.append(f"✅ Máxima sincronización en **κ = {k_opt}** (ρ_sync = {rho_max*100:.1f}%)")
        if k_opt == k_min:
            lines.append("La sincronización es mayor para κ pequeño (umbral bajo).")
        elif k_opt == k_max_df:
            lines.append("La sincronización crece con κ hasta el extremo explorado.")
        else:
            lines.append(f"Existe un κ óptimo interior: κ = {k_opt}.")

    return "\n\n".join(lines)


# ============================================================
# 7. Condición inicial
# ============================================================

def construir_theta0(N, M, modo):
    if modo == "Ejemplo paper":
        base  = [0, M // 2, M - 1]
        theta = (base + [0] * N)[:N]
        return np.array(theta, dtype=int) % M
    elif modo == "Manual simple":
        return np.array([i % M for i in range(N)], dtype=int)
    else:
        return np.random.randint(0, M, size=N)


# ============================================================
# 8. Interfaz Streamlit
# ============================================================

st.title("Caminos hacia la sincronización en una red de autómatas de Kuramoto discreta sobre grafos")
st.markdown("##### Edgardo Ugalde · Alejandro Barrientos")

st.markdown("**Red de autómatas celulares** sobre $G=(V,E)$ con alfabeto $\\mathbb{Z}_M$:")
st.latex(r"F: \mathbb{Z}_M^V \to \mathbb{Z}_M^V")
st.markdown("La trayectoria $\\theta^0, \\theta^1, \\theta^2, \\dots$ busca alcanzar la diagonal discreta:")
st.latex(r"\Delta_{\mathbb{Z}_M} = \{\, \theta \in \mathbb{Z}_M^V : \theta_i = \theta_j \ \forall\, i,j \in V \,\}")

st.markdown("---")
st.markdown("**Red de Kuramoto discreta** donde:")
st.latex(r"S_i(\theta) = \sum_{j \in N(i)} d_M(\theta_j,\, \theta_i)")
st.markdown("Donde:")
st.latex(r"d_M(a,b) = \left(\!\left(a - b + \left\lfloor\tfrac{M}{2}\right\rfloor\right)\bmod M\right) - \left\lfloor\tfrac{M}{2}\right\rfloor")
st.markdown("Donde:")
st.latex(r"""
\sigma_\kappa(S) = \begin{cases}
+1 & S \geq \kappa \\
-1 & S \leq -\kappa \\
0  & |S| < \kappa
\end{cases}
""")
st.markdown("El mapa $F_\\kappa : \\mathbb{Z}_M^V \\to \\mathbb{Z}_M^V$ actúa como:")
st.latex(r"(F_\kappa(\theta))_i = \theta_i + \omega^* + \sigma_\kappa(S_i(\theta)) \pmod{M}")
st.markdown("con $\\omega^* \\in \\mathbb{Z}_M$ frecuencia homogénea y $\\kappa \\in \\mathbb{Z}_{\\geq 1}$ umbral entero.")



st.sidebar.header("⚙️ Parámetros")

N     = st.sidebar.slider("N (vértices)", 2, 7, 3)
M     = st.sidebar.slider("M (fases)", 2, 12, 4)

# Calcular kappa_max dinámicamente
_modelo_ref = KuramotoDiscreto(N=N, M=M, kappa=1, tipo_grafo="Completo")
_degmax     = max(dict(_modelo_ref.G.degree()).values())
_kappa_max  = max(1, _degmax * (M // 2) + 1)

kappa = st.sidebar.slider("κ (kappa)", 1, _kappa_max, min(2, _kappa_max))
omega = st.sidebar.slider("ω* (omega)", 0, M - 1, 0)
epsilon     = st.sidebar.slider("ε (epsilon)", 0.1, float(M // 2 + 1), 1.0, step=0.1)
T           = st.sidebar.slider("T (pasos)", 1, 100, 20)
tipo_grafo  = st.sidebar.selectbox("Tipo de grafo", ["Completo", "Camino", "Ciclo", "Estrella"])
modo_theta  = st.sidebar.selectbox("Condición inicial θ₀", ["Ejemplo paper", "Aleatoria", "Manual simple"])
epsilon_modo = st.sidebar.radio(
    "Subred ε-sincronizada",
    ["estricto", "incluyente"],
    format_func=lambda x: "Estricto d < ε (paper)" if x == "estricto" else "Incluyente d ≤ ε (didáctico)",
)

st.sidebar.markdown("---")

# ── Tabs principales ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🧪 Simulación individual", "🔍 Exploración exhaustiva", "📊 Barrido de κ", "🖼️ Imágenes del proyecto"])


# ────────────────────────────────────────────────────────────
# TAB 1 — Simulación individual
# ────────────────────────────────────────────────────────────
with tab1:
    if st.button("▶️ Simular trayectoria", type="primary"):
        modelo  = KuramotoDiscreto(N=N, M=M, kappa=kappa, omega=omega,
                                   tipo_grafo=tipo_grafo, epsilon=epsilon,
                                   epsilon_modo=epsilon_modo)
        theta0   = construir_theta0(N, M, modo_theta)
        resultado = modelo.simular(theta0, T=T)
        clasif    = modelo.clasificar_atractor(theta0, T_max=max(200, T))

        st.subheader("Parámetros usados")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Grafo", tipo_grafo)
        col2.metric("θ₀", str(tuple(theta0)))
        col3.metric("Tipo atractor", clasif["tipo"])
        col4.metric("r final", f"{resultado['r_hist'][-1]:.4f}")

        st.subheader("Visualizaciones")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_heatmap(resultado))
        with c2:
            st.pyplot(fig_order_parameter(resultado))

        st.subheader("Snapshots del grafo")
        mitad = len(resultado["trayectoria"]) // 2
        g1, g2, g3 = st.columns(3)
        with g1:
            st.pyplot(fig_grafo(modelo, resultado["trayectoria"][0], "Snapshot inicial"))
        with g2:
            st.pyplot(fig_grafo(modelo, resultado["trayectoria"][mitad], f"Snapshot t={mitad}"))
        with g3:
            st.pyplot(fig_grafo(modelo, resultado["trayectoria"][-1], f"Snapshot final t={T}"))

        st.subheader("Tabla de trayectoria")
        st.dataframe(modelo.tabla_trayectoria(resultado), use_container_width=True)

        st.subheader("Conclusión")
        st.info(conclusion_simulacion(modelo, resultado, clasif, T))


# ────────────────────────────────────────────────────────────
# TAB 2 — Exploración exhaustiva
# ────────────────────────────────────────────────────────────
with tab2:
    total = M ** N
    st.write(f"**Espacio total:** M^N = {M}^{N} = {total} configuraciones")

    if total > 20_000:
        st.warning("⚠️ Espacio demasiado grande (> 20 000). Reduce N o M.")
    else:
        if st.button("🔍 Explorar todo el espacio", type="primary"):
            with st.spinner("Explorando configuraciones..."):
                res = exploracion_exhaustiva(N=N, M=M, kappa=kappa, omega=omega,
                                             tipo_grafo=tipo_grafo, epsilon=epsilon,
                                             epsilon_modo=epsilon_modo, T_max=150)

            st.subheader("Resumen")
            df_res = pd.DataFrame([{
                "total":        res["total_configuraciones"],
                "sync":         res["sync"],
                "fix":          res["fix"],
                "cyc":          res["cyc"],
                "transitorio":  res["transitorio"],
                "rho_sync":     round(res["rho_sync"], 4),
                "rho_fix":      round(res["rho_fix"], 4),
                "rho_cyc":      round(res["rho_cyc"], 4),
                "rho_trans":    round(res["rho_transitorio"], 4),
                "subredes":     res["num_subredes_realizables"],
                "transiciones": res["num_transiciones"],
                "Bell_BN":      res["Bell_BN"],
            }])
            st.dataframe(df_res, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if res["tiempos_sync"]:
                    st.pyplot(fig_hist_tiempos(res["tiempos_sync"]))
                else:
                    st.info("No hubo trayectorias sincronizadas.")
            with c2:
                fig_dt = fig_diagrama_transiciones(res["transiciones"])
                if fig_dt:
                    st.pyplot(fig_dt)
                else:
                    st.info("No hay transiciones no triviales entre subredes.")

            st.subheader("Conclusión")
            st.info(conclusion_exhaustiva(res, N, M, kappa, tipo_grafo))


# ────────────────────────────────────────────────────────────
# TAB 3 — Barrido de kappa
# ────────────────────────────────────────────────────────────
with tab3:
    total = M ** N
    st.write(f"**Configuraciones por κ:** {total}")

    if total > 10_000:
        st.warning("⚠️ Barrido pesado. Reduce N o M antes de ejecutar.")
    else:
        if st.button("📊 Ejecutar barrido de κ", type="primary"):
            with st.spinner("Barriendo valores de κ..."):
                df_barrido = barrido_kappa(N=N, M=M, omega=omega,
                                           tipo_grafo=tipo_grafo, epsilon=epsilon,
                                           epsilon_modo=epsilon_modo, T_max=150)

            st.subheader("Tabla de resultados")
            st.dataframe(df_barrido, use_container_width=True)

            st.subheader("Mapa de regímenes")
            st.pyplot(fig_mapa_regimenes(df_barrido))

            st.subheader("Conclusión")
            st.info(conclusion_barrido(df_barrido))


# ────────────────────────────────────────────────────────────
# TAB 4 — Galería de imágenes explicativas
# ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🖼️ Imágenes explicativas del proyecto")

    BASE = os.path.dirname(os.path.abspath(__file__))

    imagenes = [
        {"ruta": os.path.join(BASE, "imagenes", "imagen1.jpg"), "titulo": "Imagen 1"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen2.jpg"), "titulo": "Imagen 2"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen3.jpg"), "titulo": "Imagen 3"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen4.jpg"), "titulo": "Imagen 4"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen5.jpg"), "titulo": "Imagen 5"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen6.jpg"), "titulo": "Imagen 6"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen7.jpg"), "titulo": "Imagen 7"},
        {"ruta": os.path.join(BASE, "imagenes", "imagen8.jpg"), "titulo": "Imagen 8"},
    ]

    for img in imagenes:
        st.image(img["ruta"], caption=img["titulo"], use_column_width=True)
        st.divider()

