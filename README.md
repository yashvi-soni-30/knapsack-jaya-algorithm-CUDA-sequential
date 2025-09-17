# 🧮 Jaya Algorithm for 0/1 Knapsack (Sequential + CUDA)  
**License:** MIT · **Language:** C++ & CUDA  

A high-performance implementation of the **Jaya Algorithm** applied to the **0/1 Knapsack Problem**, featuring both **serial (CPU)** and **parallel (CUDA GPU)** versions. The project demonstrates significant improvements in **execution time** and **solution quality** using GPU parallelization.  

---

## ⚙️ Problem Overview  

The **0/1 Knapsack Problem**:  
- Given `n` items, each with a **weight** `w[i]` and a **profit** `p[i]`,  
- Select items such that:  

\[
\text{maximize } \sum p[i]x[i], \quad \text{subject to } \sum w[i]x[i] \leq C
\]  

where `x[i] ∈ {0,1}` and `C` is the knapsack capacity.  

---

## 🧠 Algorithm Overview  

### Jaya Algorithm  
- **Population-based** metaheuristic with no algorithm-specific parameters.  
- Moves candidate solutions closer to the **best solution** and away from the **worst solution**.  
- Iteratively updates population until convergence or iteration limit.  

### Hybrid Design (CPU + GPU)  
- **Sequential version (CPU C++)**: Baseline implementation.  
- **CUDA version (GPU)**: Parallel evaluation of population for significant speedup.  

---

## 📊 Performance Results  

| Version   | Execution Time | Best Profit | Speedup | Solution Quality |
|-----------|----------------|-------------|---------|------------------|
| Serial    | 15.42s         | 442         | 1.0x    | Good             |
| CUDA      | 3.12s          | 448         | 4.9x    | Better           |

**Key Achievements:**  
- ⚡ ~5× performance improvement with CUDA parallelization  
- 🎯 Slightly better optimization quality on large instances  
- 🔄 100,000 iterations with 500 population size under 4 seconds  

---
## 🏗️ Repository Structure  

├── src/
│ ├── jaya_sequential.cpp # Sequential CPU implementation
│ ├── jaya_cuda.cu # CUDA GPU implementation
│ ├── utils.h # Common utilities
│ └── utils.cpp # Helper functions
│
├── data/
│ └── instances/ # Knapsack problem instances
│
├── results/
│ ├── seq_results.txt # Serial execution results
│ ├── cuda_results.txt # CUDA execution results
│ └── comparison.csv # Performance comparison
│
├── notebooks/
│ └── Jaya_Knapsack.ipynb # Google Colab notebook (ready-to-run)
│
├── docs/
│ ├── algorithm.md # Detailed explanation of Jaya
│ └── performance.md # Performance graphs and analysis
│
├── Makefile # Build configuration
├── requirements.txt # Python deps for plotting (matplotlib, pandas)
└── README.md # This file

yaml
Copy code

---

## 🚀 Setup & Execution  

### Run on Google Colab  
1. Open `notebooks/Jaya_Knapsack.ipynb` in Colab.  
2. Enable **GPU runtime**: `Runtime > Change runtime type > GPU`.  
3. Run cells to:  
   - Compile CPU and GPU versions  
   - Execute on sample datasets  
   - Record performance and plot graphs  

### Manual Build (if running locally with CUDA)  

```bash
# CPU version
g++ src/jaya_sequential.cpp src/utils.cpp -o bin/jaya_seq -O3

# GPU version
nvcc src/jaya_cuda.cu src/utils.cpp -o bin/jaya_cuda -O3 -arch=sm_60
Run:

bash
Copy code
./bin/jaya_seq data/instances/input1.txt
./bin/jaya_cuda data/instances/input1.txt
🔬 Algorithm Details
Jaya Update Rule
cpp
Copy code
for (int i = 0; i < dim; i++) {
    double r1 = random(0,1), r2 = random(0,1);
    newSol[i] = sol[i] 
                + r1 * (best[i] - abs(sol[i])) 
                - r2 * (worst[i] - abs(sol[i]));
}
Moves closer to best solution

Moves away from worst solution

No control parameters required

CUDA Parallelization
Block size: 256 threads

Each thread: evaluates fitness of one candidate solution

Shared memory used for local reductions

📌 Future Work
Add comparisons with Genetic Algorithm, PSO, and RAO

Use multi-GPU execution for very large instances

Explore hybrid Jaya + GA approach for knapsack

👨‍💻 Author
Yashvi Soni
📧 soniyashvi3142@gmail.com


## 🏗️ Repository Structure  

