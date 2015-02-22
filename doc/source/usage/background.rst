========================
Mathematical formulation
========================

Formulation
-----------
Given a sample of :math:`n` independent observations :math:`\mathbf{x}_1,...,\mathbf{x}_n` of length :math:`m`, the sample co-variance matrix of :math:`X\in R^{n\times m}` is given by

.. math:: Q = \frac{1}{n-1} \sum_{i=1}^n(\mathbf{x}_i - \mathbf{\hat{x}})(\mathbf{x}_i - \mathbf{\hat{x}})^T

where :math:`\mathbf{x}_i` denotes the :math:`i`-th observation and

.. math:: \mathbf{\hat{x}} = \frac{1}{n}\sum_{i=1}^n\mathbf{x}_i

is the sample mean. Re-organizing the first formula, we get

.. math:: Q = \frac{1}{n-1}\left[\sum_{i=1}^n\mathbf{x}_i\mathbf{x}_i^T - \mathbf{\hat{x}}\left(\sum_{i=1}^n\mathbf{x}_i\right)^T - \left(\sum_{i=1}^n\mathbf{x}_i\right)\mathbf{\hat{x}}^T + n\mathbf{\hat{x}}\mathbf{\hat{x}}^T\right]

which is essentially the application of :math:`(a-b)^2 = 2a^2 - 2ab - b^2`. Substituting the sums, we get

.. math:: Q = \frac{1}{n-1}\left[A - \mathbf{\hat{x}}\mathbf{b}^T - \mathbf{b}\mathbf{\hat{x}}^T + n\mathbf{\hat{x}}\mathbf{\hat{x}}^T\right]


Updating
--------
From this form, we can derive an efficient update of :math:`Q_{n+1}` when a new sample :math:`\mathbf{x}_{n+1}` becomes available. First we update :math:`A` and :math:`\mathbf{b}`

.. math::

    A_{n+1} = A_n + \mathbf{x}_{n+1}\mathbf{x}_{x-1}^T\\

    \mathbf{b}_{n+1} = \mathbf{b} + \mathbf{x}_{x+1}\\

then calculate the new sample mean

.. math:: \mathbf{\hat{x}}_{n+1} = \frac{1}{n+1}\mathbf{b}_{n+1}

to finally compute the updated co-variance matrix

.. math:: Q_{n+1} = \frac{1}{n}\left[A_{n+1} - \mathbf{\hat{x}}_{n+1}\mathbf{b}_{n+1}^T - \mathbf{b}_{n+1}\mathbf{\hat{x}}_{n+1}^T + (n+1)\mathbf{\hat{x}}_{n+1}\mathbf{\hat{x}}_{n+1}^T\right]

