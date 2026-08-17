"""Performance benchmarks for the stochastic-LLE solver.

These modules MEASURE the solver; they never change it. Nothing here is imported
by ``simulator/``, ``analysis/`` or ``model/``, and no benchmark writes into
``analysis/results/`` -- benchmark artifacts live beside their scripts in
``benchmarks/`` so a timing run can never be mistaken for a physics result.

Modules
-------
runtime_scaling:
    Wall-clock and memory scaling of :func:`simulator.lle_solver.solve_lle_ssfm_jax`
    across the FFT grid ``n_tau``, the record length ``t_slow``, the trajectory
    batch ``n_traj``, the noise configuration and the JAX backend. Writes
    ``benchmarks/runtime.json`` and ``benchmarks/runtime_table.md``.
"""
