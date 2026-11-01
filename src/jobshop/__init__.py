"""Job shop (J||Cmax) layer of the neighborhood-family project (articles/jobshop/PLAN.md).

instance       JSPLIB / Taillard formats, Taillard's generator (+ route disorder rho), bounds
schedule       disjunctive-graph evaluation, critical path, blocks, starts, kick
feasibility    theorem T2 (reorderings inside critical blocks)
neighborhoods  border-move family (swap / N6 / N7 x reach k x single / compound)
algorithms     ILS and SA with the control flow and constants of the PFSP code
runner         one run -> result dict in the runner / extract_budgets.py format
"""
