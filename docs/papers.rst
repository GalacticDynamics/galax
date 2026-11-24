.. _galax-potential-papers:

*************************
Potentials from papers
*************************

:mod:`galax.potential.papers` holds potentials set up as in a published paper.
Each function is named for the paper's DOI: every character that is not a
letter or digit becomes ``_``, and the name starts with ``doi_``. The function's
docstring describes the paper's model and the parameters it takes.

.. list-table::
    :header-rows: 1

    * - Function
      - Paper
      - Returns
    * - ``doi_10_1093_mnras_stad551``
      - Koposov, Erkal, Li et al. 2023, MNRAS 521, 4936
        (`doi:10.1093/mnras/stad551 <https://doi.org/10.1093/mnras/stad551>`_,
        `arXiv:2211.04495 <https://arxiv.org/abs/2211.04495>`_)
      - Milky Way dark-matter halo: a flattened generalized NFW with a
        Gaussian cutoff and a tilted symmetry axis, as a
        ``TransformedPotential`` of ``flattened_gnfw_halo``
