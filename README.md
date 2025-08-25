# rl-project-1


# DQN para LunarLander-v3 (Gymnasium)

* **Título:** Aterrizaje autónomo con DQN en LunarLander-v3 (Gymnasium)  
* **Autor:** Freddy Julian Salas  
* **Curso:** Aprendizaje por refuerzo 1  
* **Fecha:** 24-08-2025  

---

## 1. Introducción y enunciado del problema

El objetivo es aprender una **política óptima** $\pi(a|s)$ que permita **aterrizar suavemente** el módulo lunar en el entorno **LunarLander-v3**. El agente observa un vector continuo de 8 variables de estado y selecciona una de 4 acciones discretas.

**Puntos del enunciado cubiertos**:

* Se utiliza **Gymnasium** y el environment **LunarLander-v3**.
* Incluye fragmento de código relevante.
* Se incluye un **gráfico de convergencia** (reward vs episodios) con **media móvil**.
* Se incluye la URL del repo donde esta alojado el proyecto.

---

## 2. Metodología: Deep Q-Learning (DQN)

### 2.1. Resumen de DQN

DQN aproxima la **función de acción-valor** $Q_\theta(s,a)$ con una red neuronal y aprende a minimizar el error temporal usando **replay buffer** y una **red objetivo (target network)** para estabilizar el entrenamiento.

* **Replay Buffer**: rompe la correlación temporal muestreando mini-batches i.i.d. desde una memoria de transiciones $(s,a,r,s',done)$.
* **Política** **$\epsilon$**-greedy: explota el mejor $Q$ estimado con prob. $1-\epsilon$ y explora acción aleatoria con prob. $\epsilon$. $\epsilon$ decae con los pasos.
* **Target Network (tgt)**: copia retrasada de la red online, reduce la inestabilidad al desacoplar el objetivo de entrenamiento.

### 2.2. ¿Qué es la red objetivo **tgt**?

En el script, `tgt` es la **red objetivo**. Se usa la red online `qnet` para calcular $Q_\theta(s,a)$ y **otra red fija por algunos pasos** ($\theta^-$) para el término objetivo:

$$
\underbrace{Q_\theta(s,a)}_{\text{online}} \;\; \text{vs.}\;\; \underbrace{r + \gamma\,\max_{a'} Q_{\theta^-}(s',a')}_{\text{objetivo (target)}}
$$

Si se usan los mismos parámetros para ambos términos, el objetivo **se mueve demasiado** durante el entrenamiento, generando **divergencia**. Con `tgt` el objetivo cambia **más lentamente**:

* **Actualización suave (soft):** `tgt ← τ·qnet + (1−τ)·tgt` en cada paso.

**Fragmento clave del código**:

```python
def soft_update(target_net, online_net, tau=1.0):
    """tau=1.0 -> copia dura. Si 0<tau<1, soft-update."""
    for tgt_p, src_p in zip(target_net.parameters(), online_net.parameters()):
        tgt_p.data.copy_(tau * src_p.data + (1.0 - tau) * tgt_p.data)
````

Este procedimiento suaviza la transición de parámetros en la red objetivo.

## 2.3. Arquitectura de la red

- MLP 8→128→128→4 con activaciones ReLU.  
- **Loss**: Huber (SmoothL1Loss).  
- **Optimización**: Adam.  
- **Clipping**: `clip_grad_norm_` para estabilidad.  

## 2.4. Hiperparámetros usados

- `replay_size = 100_000`  
- `seed = 42`  
- `alpha = 0.1` (learning rate)  
- `gamma = 0.99` (discount factor)  
- `epsilon = 0.2` (exploration rate inicial, sobreescrito por eps_start/eps_end)  
- `num_episodes = 1000`  
- `max_steps_per_episode = 1000`  
- `learning_rate = 1e-3`  
- `eps_start = 1.0`  
- `eps_end = 0.05`  
- `eps_decay_steps = 50_000`  
- `batch_size = 128`  
- `target_update_hard = 2000`  
- `target_update_tau = 0.005`  
- `eval_episodes = 10`  
- `eval_render_mode = "human"`  

---

## 3. Implementación y ejecución

### 3.1. Dependencias
- `gymnasium[box2d] == 0.29.1`  
- `torch`, `numpy`, `matplotlib`, `tqdm`  

### 3.2. Ejecución
- Entrenamiento: `python main.py`  

### 3.3. Artefactos
- `dqn_lunarlander.pt` (pesos)  
- `convergence.png` (convergencia + media móvil)  

---

## 4. Resultados

Durante 1000 episodios de entrenamiento:

- La recompensa inicial rondaba entre **-300 y -100**.  
- El agente fue mejorando progresivamente, alcanzando recompensas promedio positivas.  
- La curva suavizada muestra que alrededor del episodio 400–500 alcanzó picos de **+200 reward**, aunque luego hubo oscilaciones.  

**Figura 1.** Convergencia (reward vs episodios, suavizado con media móvil de ventana 50).  

Esto evidencia que el agente aprendió una política razonable, aunque aún con alta varianza.  

---

## 5. Dificultades y soluciones

- **Dependencias Box2D**: fallas de compilación en macOS → instalar CLT y `swig`.  
- **Inestabilidad del entrenamiento**: mitigada con replay grande, Huber loss, target network y clipping de gradiente.  
- **Render en Colab**: usar `render_mode="rgb_array"` y guardar frames si se requiere animación.  

---

## 6. Conclusiones

- DQN con target network permite estabilizar el aprendizaje en LunarLander.  
- La media móvil facilita visualizar la tendencia de convergencia.  
- Ajustes finos en $\epsilon$-decay, LR y frecuencia de actualización del target impactan la velocidad/calidad del aprendizaje.  

---

## 7. Fragmentos de código relevantes

- Definición de la red (MLP 8-128-128-4).  
- Bucle de entrenamiento con cálculo de la pérdida y actualización del target.  
- Generación de la figura de convergencia.  

---

## 8. Trabajo futuro

- Double DQN, Dueling DQN y Prioritized Replay.  
- Regularización y schedulers de LR.  
- Evaluaciones con semillas múltiples y reporte estadístico.  

---

## 9. Referencias

- Mnih et al., 2015. *Human-level control through deep reinforcement learning.*  
- Documentación de Gymnasium y entorno LunarLander. 
