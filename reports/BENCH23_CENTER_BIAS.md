# Center-bias test on the 23 classic functions of the SOCIAL paper

Median final error (f - f*) over seeds; lower is better. Rank 1 = best of the six methods.

|                                 |   SOCIAL (repo, 60 agents) |   SOCIAL (paper Alg. 1) |    CMA-ES |         DE |        PSO |     random |
|:--------------------------------|---------------------------:|------------------------:|----------:|-----------:|-----------:|-----------:|
| ('Ackley', 'original')          |                   6.45     |                4.66     |  2.05e-09 |   8.95     |   3.61     |  19.7      |
| ('Ackley', 'shifted')           |                  19.5      |               19.9      |  1.82e-09 |  10.2      |  15.7      |  20.6      |
| ('Branin', 'original')          |                  -0.000113 |               -0.000113 | -0.000113 |  -0.000113 |  -0.000113 |   0.00432  |
| ('Camel-Back', 'original')      |                  -2.85e-05 |               -2.84e-05 | -2.85e-05 |  -2.85e-05 |  -2.85e-05 |   0.00835  |
| ('Foxholes', 'original')        |                  -0.526    |               -0.526    | -0.526    |  -0.526    |  -0.526    |  -0.464    |
| ('Goldstein-Price', 'original') |                   5.05e-09 |                4.31e-07 | -6.37e-14 |  -6.39e-14 |  -6.39e-14 |   0.0873   |
| ('Griewank', 'original')        |                   1.07     |                1.03     |  3.11e-14 |   4.09     |   0.0111   | 368        |
| ('Griewank', 'shifted')         |                  33.6      |                1.11     |  3.34e-14 |   7.63     | 116        | 572        |
| ('Hartman', 'original')         |                   0.00847  |               -2.94e-05 | -0.00278  |  -0.00278  |  -0.00278  |   0.00651  |
| ('Kowalik', 'original')         |                   0.00188  |                0.00188  |  0.0018   |   0.00183  |   0.0029   |   0.00728  |
| ('Penalized', 'original')       |                   2.87     |                9.68     |  3.45e-14 |   2.37e+04 |   1.28e+06 |   9.8e+06  |
| ('Penalized', 'shifted')        |                   2.76e+05 |               57.8      |  3.82e-14 |   1.35e+06 |   2.61e+07 |   6.56e+07 |
| ('Penalized2', 'original')      |                  10        |                3.04     |  3.06e-14 |   1.7e+03  |   0.859    |   1e+06    |
| ('Penalized2', 'shifted')       |                   3.54e+04 |               78.3      |  2.15e-14 |   6.33e+04 |   1.9e+05  |   4.85e+06 |
| ('Quartic', 'original')         |                   3.68e-05 |                2.31e-06 |  2.31e-16 |   0.165    |   1.91e-31 |  43.6      |
| ('Quartic', 'shifted')          |                   1.9      |                0.000343 |  2.12e-16 |   1.22     |  13.8      | 140        |
| ('Rastrigin', 'original')       |                  22.9      |               38.2      | 22.9      |  33.1      | 109        | 339        |
| ('Rastrigin', 'shifted')        |                 227        |              268        | 67.7      |  66.9      |  97.7      | 431        |
| ('Rosenbrock', 'original')      |                 621        |              393        |  2.11     |   1.78e+05 | 142        |   1e+08    |
| ('Rosenbrock', 'shifted')       |                   3.52e+06 |                3.4e+03  |  1.53     |   2.32e+06 |   2.24e+07 |   3.16e+08 |
| ('Schwefel_1_2', 'original')    |                 589        |              834        |  3.16e-14 |   1.04e+03 |   1.08e+04 |   4.54e+04 |
| ('Schwefel_1_2', 'shifted')     |                   2.19e+04 |                8.01e+03 |  3.59e-14 |   4.88e+03 |   2.41e+04 |   9.41e+04 |
| ('Schwefel_2_21', 'original')   |                   8.08     |               14.2      |  4.82e-09 |  34.5      |   4.57     |  70.6      |
| ('Schwefel_2_21', 'shifted')    |                  62.5      |               59.5      |  4.56e-09 |  72        |  40        | 101        |
| ('Schwefel_2_22', 'original')   |                   6.02     |                4.61     |  3.26e-09 |   1.23     |  10        |   4.78e+04 |
| ('Schwefel_2_22', 'shifted')    |                 130        |              347        |  3.56e-09 |   0.971    |  32        |   1.01e+06 |
| ('Schwefel_2_26', 'original')   |                  -1.94e+04 |               -1.88e+04 | -2.23e+04 |  -2.18e+04 |  -2.08e+04 |  -1.68e+04 |
| ('Schwefel_2_26', 'shifted')    |                  -2.17e+04 |               -2.19e+04 | -2.71e+04 |  -2.74e+04 |  -2.45e+04 |  -1.91e+04 |
| ('Shekel1', 'original')         |                   0.068    |               -0.000344 | -0.000368 |   0.0607   |  -0.000368 |   0.327    |
| ('Shekel2', 'original')         |                  -0.0307   |               -0.0294   | -0.0308   |   0.022    |  -0.00197  |   6.27     |
| ('Shekel3', 'original')         |                   0.14     |                0.068    | -0.0136   |   0.18     |   0.0714   |   6.03     |
| ('Shekel4', 'original')         |                   0.161    |                0.161    | -0.0165   |   0.129    |   0.0998   |   5.56     |
| ('Sphere', 'original')          |                   7.72     |                5.91     |  2.16e-14 | 372        |   1.83e-14 |   4.07e+04 |
| ('Sphere', 'shifted')           |                   3.28e+03 |                8.59     |  2.73e-14 |   2.19e+03 |   1.52e+04 |   6.72e+04 |
| ('Step', 'original')            |                   0        |               47.5      |  0        |  46        |  10        |   4.08e+04 |
| ('Step', 'shifted')             |                   5.78e+03 |               95        |  0        | 795        |   1.36e+04 |   6.51e+04 |

## Rank of each method per function and version

|                                 |   SOCIAL (repo, 60 agents) |   SOCIAL (paper Alg. 1) |   CMA-ES |   DE |   PSO |   random |
|:--------------------------------|---------------------------:|------------------------:|---------:|-----:|------:|---------:|
| ('Ackley', 'original')          |                        4.0 |                     3.0 |      1.0 |  5.0 |   2.0 |      6.0 |
| ('Ackley', 'shifted')           |                        4.0 |                     5.0 |      1.0 |  2.0 |   3.0 |      6.0 |
| ('Branin', 'original')          |                        4.0 |                     5.0 |      2.0 |  2.0 |   2.0 |      6.0 |
| ('Camel-Back', 'original')      |                        4.0 |                     5.0 |      3.0 |  1.5 |   1.5 |      6.0 |
| ('Foxholes', 'original')        |                        4.0 |                     5.0 |      3.0 |  2.0 |   1.0 |      6.0 |
| ('Goldstein-Price', 'original') |                        4.0 |                     5.0 |      3.0 |  1.5 |   1.5 |      6.0 |
| ('Griewank', 'original')        |                        4.0 |                     3.0 |      1.0 |  5.0 |   2.0 |      6.0 |
| ('Griewank', 'shifted')         |                        4.0 |                     2.0 |      1.0 |  3.0 |   5.0 |      6.0 |
| ('Hartman', 'original')         |                        6.0 |                     4.0 |      3.0 |  1.5 |   1.5 |      5.0 |
| ('Kowalik', 'original')         |                        4.0 |                     3.0 |      1.0 |  2.0 |   5.0 |      6.0 |
| ('Penalized', 'original')       |                        2.0 |                     3.0 |      1.0 |  4.0 |   5.0 |      6.0 |
| ('Penalized', 'shifted')        |                        3.0 |                     2.0 |      1.0 |  4.0 |   5.0 |      6.0 |
| ('Penalized2', 'original')      |                        4.0 |                     3.0 |      1.0 |  5.0 |   2.0 |      6.0 |
| ('Penalized2', 'shifted')       |                        3.0 |                     2.0 |      1.0 |  4.0 |   5.0 |      6.0 |
| ('Quartic', 'original')         |                        4.0 |                     3.0 |      2.0 |  5.0 |   1.0 |      6.0 |
| ('Quartic', 'shifted')          |                        4.0 |                     2.0 |      1.0 |  3.0 |   5.0 |      6.0 |
| ('Rastrigin', 'original')       |                        1.0 |                     4.0 |      2.0 |  3.0 |   5.0 |      6.0 |
| ('Rastrigin', 'shifted')        |                        4.0 |                     5.0 |      2.0 |  1.0 |   3.0 |      6.0 |
| ('Rosenbrock', 'original')      |                        4.0 |                     3.0 |      1.0 |  5.0 |   2.0 |      6.0 |
| ('Rosenbrock', 'shifted')       |                        4.0 |                     2.0 |      1.0 |  3.0 |   5.0 |      6.0 |
| ('Schwefel_1_2', 'original')    |                        2.0 |                     3.0 |      1.0 |  4.0 |   5.0 |      6.0 |
| ('Schwefel_1_2', 'shifted')     |                        4.0 |                     3.0 |      1.0 |  2.0 |   5.0 |      6.0 |
| ('Schwefel_2_21', 'original')   |                        3.0 |                     4.0 |      1.0 |  5.0 |   2.0 |      6.0 |
| ('Schwefel_2_21', 'shifted')    |                        4.0 |                     3.0 |      1.0 |  5.0 |   2.0 |      6.0 |
| ('Schwefel_2_22', 'original')   |                        4.0 |                     3.0 |      1.0 |  2.0 |   5.0 |      6.0 |
| ('Schwefel_2_22', 'shifted')    |                        4.0 |                     5.0 |      1.0 |  2.0 |   3.0 |      6.0 |
| ('Schwefel_2_26', 'original')   |                        4.0 |                     5.0 |      1.0 |  2.0 |   3.0 |      6.0 |
| ('Schwefel_2_26', 'shifted')    |                        5.0 |                     4.0 |      2.0 |  1.0 |   3.0 |      6.0 |
| ('Shekel1', 'original')         |                        5.0 |                     3.0 |      2.0 |  4.0 |   1.0 |      6.0 |
| ('Shekel2', 'original')         |                        2.0 |                     3.0 |      1.0 |  5.0 |   4.0 |      6.0 |
| ('Shekel3', 'original')         |                        4.0 |                     2.0 |      1.0 |  5.0 |   3.0 |      6.0 |
| ('Shekel4', 'original')         |                        4.0 |                     5.0 |      1.0 |  3.0 |   2.0 |      6.0 |
| ('Sphere', 'original')          |                        4.0 |                     3.0 |      2.0 |  5.0 |   1.0 |      6.0 |
| ('Sphere', 'shifted')           |                        4.0 |                     2.0 |      1.0 |  3.0 |   5.0 |      6.0 |
| ('Step', 'original')            |                        1.5 |                     5.0 |      1.5 |  4.0 |   3.0 |      6.0 |
| ('Step', 'shifted')             |                        4.0 |                     2.0 |      1.0 |  3.0 |   5.0 |      6.0 |

- F1-F13, original: mean rank SOCIAL (repo, 60 agents) 3.19, SOCIAL (paper Alg. 1) 3.46, CMA-ES 1.27, DE 4.15, PSO 2.92, random 6.00
- F1-F13, shifted: mean rank SOCIAL (repo, 60 agents) 3.92, SOCIAL (paper Alg. 1) 3.00, CMA-ES 1.15, DE 2.77, PSO 4.15, random 6.00
- F14-F23 (as published): mean rank SOCIAL (repo, 60 agents) 4.10, SOCIAL (paper Alg. 1) 4.00, CMA-ES 2.00, DE 2.75, PSO 2.25, random 5.90
