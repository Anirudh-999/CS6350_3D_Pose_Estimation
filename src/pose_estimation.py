import numpy as np
from math import atan, degrees, radians, tan, sin, cos

def compute_angles(beta1_deg, coeffs):
    """
    Compute α1, β2, α2, β1' (new beta1) using weak-perspective relationships.
    Equations correspond to (19)–(22) in Mao et al. (2007).
    """
    beta1 = radians(beta1_deg)
    N, O, P, Q = coeffs["N"], coeffs["O"], coeffs["P"], coeffs["Q"]
    a, f = sin(beta1), cos(beta1)

    # Eq. (19)
    alpha1 = np.degrees(np.arctan((a * N + f * Q) / (f * Q - a * N + 1e-8)))
    # Eq. (20)
    beta2 = np.degrees(np.arccos(np.clip(P / (f * f + 1e-8), -1, 1)))
    # Eq. (21)
    alpha2 = np.degrees(np.arctan((a * O + f * Q) / (f * O - a * Q + 1e-8)))
    # Eq. (22)
    beta1_new = np.degrees(np.arctan((sin(beta1) * Q + cos(beta1) * N) /
                                     (cos(beta1) * Q - sin(beta1) * N + 1e-8)))
    return alpha1, beta2, alpha2, beta1_new


def correct_signs_using_E(best, points_img1, points_img2):
    """
    Determines correct rotation direction using the 4th point E.
    If the relative direction of E between views is reversed, flip the signs.
    """
    alpha1, beta1, gamma1, alpha2, beta2, gamma2 = best

    # Compute midpoints of A and B (D)
    D1 = ((points_img1['A'][0] + points_img1['B'][0]) / 2,
          (points_img1['A'][1] + points_img1['B'][1]) / 2)
    D2 = ((points_img2['A'][0] + points_img2['B'][0]) / 2,
          (points_img2['A'][1] + points_img2['B'][1]) / 2)

    # Direction vectors from D → E
    vE1 = np.array(points_img1['E']) - np.array(D1)
    vE2 = np.array(points_img2['E']) - np.array(D2)

    # Flip sign if direction inverted
    if np.dot(vE1, vE2) < 0:
        alpha1, beta1, gamma1 = -alpha1, -beta1, -gamma1
        alpha2, beta2, gamma2 = -alpha2, -beta2, -gamma2
    return (alpha1, beta1, gamma1, alpha2, beta2, gamma2)

def _compute_coefficients(points_img1, points_img2):
    """Compute geometric coefficients N, O, P, Q from feature points."""
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def slope(p1, p2):
        return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-8)

    D1 = midpoint(points_img1['A'], points_img1['B'])
    D2 = midpoint(points_img2['A'], points_img2['B'])

    # γ (roll angles)
    L = slope(points_img1['A'], points_img1['B'])
    M = slope(points_img2['A'], points_img2['B'])
    gamma1, gamma2 = degrees(atan(L)), degrees(atan(M))

    # Coefficients for analytical equations
    N = slope(points_img1['C'], D1)
    O = slope(points_img2['C'], D2)
    P = (points_img2['B'][0] - points_img2['A'][0]) / (points_img1['B'][0] - points_img1['A'][0] + 1e-8)
    Q = (N + O) / 2  # Approximation

    coeffs = {"N": N, "O": O, "P": P, "Q": Q, "gamma1": gamma1, "gamma2": gamma2}
    return coeffs



def estimate_pose_from_points(points_img1, points_img2):
    """
    Estimate pose difference (Δα, Δβ, Δγ) between two images using 4 feature points.
    Points should be dictionaries containing A, B, C, E.
    """
    coeffs = _compute_coefficients(points_img1, points_img2)
    gamma1, gamma2 = coeffs["gamma1"], coeffs["gamma2"]

    # Scan β1 and minimize reconstruction error
    min_error = float('inf')
    best = None

    for beta1 in np.arange(-60, 60, 0.2):
        alpha1, beta2, alpha2, beta1_new = compute_angles(beta1, coeffs)
        error = (beta1_new - beta1)**2
        if error < min_error:
            min_error = error
            best = (alpha1, beta1, gamma1, alpha2, beta2, gamma2)

    # Fix possible sign ambiguity
    best = correct_signs_using_E(best, points_img1, points_img2)

    # Compute Δpose
    delta_pose = (
        best[3] - best[0],  # Δα
        best[4] - best[1],  # Δβ
        best[5] - best[2]   # Δγ
    )

    return delta_pose, best

