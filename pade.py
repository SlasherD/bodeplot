def pade(T, n=1, numdeg=None):
    """
    Create a linear system that approximates a delay.

    Return the numerator and denominator coefficients of the Pade approximation.

    Parameters
    ----------
    T : number
        time delay
    n : positive integer
        degree of denominator of approximation
    numdeg: integer, or None (the default)
            If None, numerator degree equals denominator degree
            If >= 0, specifies degree of numerator
            If < 0, numerator degree is n+numdeg

    Returns
    -------
    num, den : array
        Polynomial coefficients of the delay model, in descending powers of s.

    Notes
    -----
    Based on:
      1. Algorithm 11.3.1 in Golub and van Loan, "Matrix Computation" 3rd.
         Ed. pp. 572-574
      2. M. Vajta, "Some remarks on Padé-approximations",
         3rd TEMPUS-INTCOM Symposium
    
    Note:
      Changed returned coefficients to be more suitable for bode plotting ~ Dyan
    """
    if not numdeg:
        numdeg = n
    elif numdeg < 0:
        numdeg += n

    if not T >= 0:
        raise ValueError("require T >= 0")
    if not n >= 0:
        raise ValueError("require n >= 0")
    if not (0 <= numdeg <= n):
        raise ValueError("require 0 <= numdeg <= n")

    if T == 0:
        num = [1,]
        den = [1,]
    else:
        num = [0. for i in range(numdeg+1)]
        num[-1] = 1.
        cn = 1.
        for k in range(1, numdeg+1):
            # derived from Gloub and van Loan eq. for Dpq(z) on p. 572
            # this accumulative style follows Alg 11.3.1
            cn *= -T * (numdeg - k + 1)/(numdeg + n - k + 1)/k
            num[numdeg-k] = cn

        den = [0. for i in range(n+1)]
        den[-1] = 1.
        cd = 1.
        for k in range(1, n+1):
            # see cn above
            cd *= T * (n - k + 1)/(numdeg + n - k + 1)/k
            den[n-k] = cd
    return num, den
