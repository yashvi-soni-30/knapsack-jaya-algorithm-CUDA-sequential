%%writefile knapsack_jayaS.cu

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Objective: maximize total value while staying within capacity
float objectiveFunc(const vector<int>& solution, const vector<int>& weights, const vector<int>& values, int capacity) {
    int totalWeight = 0, totalValue = 0;
    for (int i = 0; i < solution.size(); i++) {
        if (solution[i]) {
            totalWeight += weights[i];
            totalValue += values[i];
        }
    }
    if (totalWeight > capacity) return 0;  // Penalize overweight solutions
    return totalValue;
}

// Sigmoid function to map values to probability
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

int main() {
    srand(time(0));
    auto start = high_resolution_clock::now();

    int dim = 200;               // Number of items
    int numParticles = 1000;      // Population size
    int numItr = 100;           // Number of iterations
    float mutationRate = 0.03f; // Mutation rate
    int capacity = 28;          // Knapsack capacity

    // Generate random weights and values for items
    vector<int> weights(dim), values(dim);
    for (int i = 0; i < dim; i++) {
        weights[i] = rand() % 10 + 1;   // 1 to 10
        values[i] = rand() % 20 + 1;    // 1 to 20
    }

    // Print item list
    cout << "Items (Weight, Value):\n";
    for (int i = 0; i < dim; i++) {
        cout << "Item " << i << ": (" << weights[i] << ", " << values[i] << ")\n";
    }
    cout << "Knapsack Capacity: " << capacity << "\n\n";

    // Step 1: Initialize binary population
    vector<vector<int>> population(numParticles, vector<int>(dim));
    for (int i = 0; i < numParticles; i++)
        for (int j = 0; j < dim; j++)
            population[i][j] = rand() % 2;


    vector<int> best(dim), worst(dim);
    float bestFit = -1.0f, worstFit = -1.0f;

    for (int itr = 0; itr < numItr; itr++) {
        bestFit = -1.0f;
        worstFit = -1.0f;

        // Step 2: Evaluate population
        for (const auto& sol : population) {
            float fit = objectiveFunc(sol, weights, values, capacity);
            if (fit > bestFit&& fit>0) {
                bestFit = fit;
                best = sol;
            }
            if ((fit < worstFit || worstFit == -1.0f)&& fit>0) {
                worstFit = fit;
                worst = sol;
            }
        }

        if (bestFit == -1.0f) {
    // All individuals were invalid; randomly select one as fallback
    best = population[rand() % numParticles];
    bestFit = objectiveFunc(best, weights, values, capacity);
}

if (worstFit == -1.0f) {
    worst = population[rand() % numParticles];
    worstFit = objectiveFunc(worst, weights, values, capacity);
}


        // Step 3: Jaya update for each particle
        for (int i = 0; i < numParticles; i++) {
            vector<int> newSol = population[i];
            for (int j = 0; j < dim; j++) {
                int x = population[i][j];
                int x_best = best[j];
                int x_worst = worst[j];

                float r1 = static_cast<float>(rand()) / RAND_MAX;
                float r2 = static_cast<float>(rand()) / RAND_MAX;

                float v = r1 * (x_best - x) - r2 * (x_worst - x);
                float p = sigmoid(v);
                float randVal = static_cast<float>(rand()) / RAND_MAX;

                newSol[j] = (randVal < p) ? 1 : 0;
            }

            // Step 4: Mutation
            for (int j = 0; j < dim; j++) {
                if (static_cast<float>(rand()) / RAND_MAX < mutationRate)
                    newSol[j] = 1 - newSol[j]; // Flip the bit
            }

            // Step 5: Replace if better
            if (objectiveFunc(newSol, weights, values, capacity) >
                objectiveFunc(population[i], weights, values, capacity)) {
                population[i] = newSol;
            }
        }

        // Output iteration stats
        cout << "Iteration " << itr + 1
             << ": Best Fitness = " << bestFit
             << ", Worst Fitness = " << worstFit << "\n";
    }

    // Final best solution

     cout << "\n Best Solution Found:\n";
    int totalWeight = 0, totalValue = 0;
    for (int i = 0; i < dim; i++) {
        if (best[i]) {
            cout << "Item " << i << " selected (Weight: " << weights[i]
                 << ", Value: " << values[i] << ")\n";
            totalWeight += weights[i];
            totalValue += values[i];
        }
    }
    cout << "\nTotal Weight = " << totalWeight
         << "\nTotal Value  = " << totalValue << endl;

    // Also compute total weight for clarity


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Execution Time: " << duration.count() << " ms" << endl;

    return 0;
}
