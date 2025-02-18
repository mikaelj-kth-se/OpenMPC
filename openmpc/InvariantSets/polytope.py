import numpy as np
import cdd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Polytope:
    def __init__(self, A, b):
        # Normalize the polytope inequalities to the form a^Tx <= 1
        A, b = self.normalize(A, b)
        
        # Ensure that b is a 1D array
        b = b.flatten()

        self.A = A
        self.b = b

        # Convert A and b into CDDlib format (H-representation)
        mat = np.hstack([b.reshape(-1, 1), -A])
        self.cdd_mat = cdd.matrix_from_array(mat, rep_type=cdd.RepType.INEQUALITY)
        self.poly = cdd.polyhedron_from_matrix(self.cdd_mat)

    def normalize(self, A, b):
        """
        Normalize the inequalities to the form a^T x <= 1.

        Args:
            A (numpy.ndarray): Inequality matrix of shape (m, n).
            b (numpy.ndarray): Right-hand side vector of shape (m,) or (m, 1).

        Returns:
            (numpy.ndarray, numpy.ndarray): Normalized A and b.
        """
        # Ensure b is a 1D array
        b = b.flatten()
        
        # Check if dimensions are compatible
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows, but b has {b.shape[0]} elements.")
        
        # Avoid division by zero
        b_norm = np.where(b == 0, 1, b)  # Use 1 to avoid division by zero
        A_normalized = A / b_norm[:, np.newaxis]
        b_normalized = np.ones_like(b)
        return A_normalized, b_normalized

    # Other methods (plot, projection, intersect, etc.) remain unchanged


    def __repr__(self):
        """
        Custom print method to display the H-representation of the polytope in a user-friendly format,
        ensuring that the string length (including minus sign) is the same for all entries of A, 
        and that entries of b are also equally spaced.
        """
        # Determine the maximum width for each element in A (for consistent alignment)
        A_max_len = max(len(f"{coef:.5f}") for row in self.A for coef in row)
        b_max_len = max(len(f"{bval:.5f}") for bval in self.b)

        repr_str = "Hyperplane representation of polytope\n"
        
        for i in range(self.A.shape[0]):
            if i == 0:
                repr_str += "  [["  # Start the first line with '[['
            else:
                repr_str += "   ["  # Subsequent lines with proper indentation
            
            # Format each row of A with the determined max width
            row_str = "  ".join(f"{coef:{A_max_len}.5f}" for coef in self.A[i])
            
            # Add the corresponding b value with consistent width
            if i == self.A.shape[0] - 1:
                repr_str += f"{row_str}] x <= [{self.b[i]:{b_max_len}.5f}]\n"  # Last line
            else:
                repr_str += f"{row_str}] |    [{self.b[i]:{b_max_len}.5f}]\n"
        
        return repr_str


    def get_V_representation(self):
        """
        Get the vertices (V-representation) of the polytope.
        """
        generators = cdd.copy_generators(self.poly)
        vertices = np.array(generators.array)[:, 1:]  # Skip the first column (homogenizing coordinate)
        return vertices


    def volume(self):
        """
        Compute the volume of the polytope. Handles 1D intervals separately.
        """
        vertices = self.get_V_representation()

        # If there are no vertices, the volume is zero
        if len(vertices) == 0:
            return 0.0

        # Handle 1D intervals (polytope in 1D)
        if vertices.shape[1] == 1:
            return np.max(vertices) - np.min(vertices)

        # Handle higher-dimensional polytopes (2D and above)
        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except Exception as e:
            print(f"Error computing volume: {e}")
            return 0.0
    
    
    def plot(self, ax=None, color='b', edgecolor='k', alpha=1.0, linestyle='-', showVertices=False):
        """
        Plot the polytope using Matplotlib.

        Args:
            ax: Matplotlib axis object.
            color: Fill color of the polytope.
            edgecolor: Color of the polytope edges (if different from color).
            alpha: Transparency of the fill.
            linestyle: Line style for the edges.
            showVertices: Boolean, whether to show the vertices as points.
        """
        # Get vertices for plotting
        vertices = self.get_V_representation()

        if len(vertices) == 0:
            return  # Nothing to plot if no vertices

        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]

        if ax is None:
            ax = plt.gca()

        if edgecolor is None:
            edgecolor = color  # Use same color for edges if edgecolor is not specified

        ax.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
                np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
                color=edgecolor, linestyle=linestyle)

        ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=color, alpha=alpha)

        if showVertices:
            ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'ro')


    def remove_redundancies(self):
        """
        Simplify the polytope by removing redundant constraints using matrix_canonicalize.
        """
        # Make a copy of the matrix
        mat_copy = cdd.matrix_copy(self.cdd_mat)
        
        # Canonicalize the matrix to remove redundant constraints
        cdd.matrix_canonicalize(mat_copy)

        # Update A and b after redundancy removal
        reduced_array = np.array(mat_copy.array)
        self.A = -reduced_array[:, 1:]
        self.b = reduced_array[:, 0]
        
        # Update the cdd matrix as well
        self.cdd_mat = mat_copy

    def projection(self, dims):
        """
        Project the polytope onto the subspace of 'dims' using cddlib block_elimination.
        Args:
            dims (list): Indices of the dimensions to keep.
        """
        A = self.A
        b = self.b

        # Convert A and b into the matrix required for CDD
        mat = np.hstack([b.reshape(-1, 1), -A])
        cdd_mat = cdd.matrix_from_array(mat.tolist(), rep_type=cdd.RepType.INEQUALITY)

        # We need to shift the dims because column 0 is b in [b -A]
        num_vars = A.shape[1]
        cols_to_eliminate = set(range(1, num_vars + 1)) - set(d + 1 for d in dims)

        # Perform block elimination using cddlib
        reduced_mat = cdd.block_elimination(cdd_mat, cols_to_eliminate)

        # Extract the reduced A and b from the resulting matrix
        reduced_array = np.array(reduced_mat.array)  # Access the matrix using .array
        reduced_A = -reduced_array[:, 1:]  # Take the matrix part (excluding the first column)
        reduced_b = reduced_array[:, 0]    # First column corresponds to b

        # Create a new polytope in the projected space
        projected_polytope = Polytope(reduced_A[:, dims], reduced_b)

        # Remove redundancies from the new polytope
        projected_polytope.remove_redundancies()

        return projected_polytope

    def intersect(self, other):
        """
        Compute the intersection of two polytopes (self and other).
        Returns:
            A new Polytope representing the intersection.
        """
        # Combine the inequalities from both polytopes
        combined_A = np.vstack([self.A, other.A])
        combined_b = np.concatenate([self.b, other.b])

        # Create a new polytope representing the intersection
        intersected_polytope = Polytope(combined_A, combined_b)

        # Remove redundancies from the intersection result
        intersected_polytope.remove_redundancies()

        return intersected_polytope

    def contains(self, x, tol=1e-8):
        """
        Check if a point x is inside the polytope.
        Args:
            x: A point as a numpy array.
        Returns:
            True if the point is inside the polytope, False otherwise.
        """
        # Check if Ax <= b holds for the point x
        return np.all(np.dot(self.A, x) <= self.b + tol)


    
    def __eq__(self, other, tol=1e-7):
        """
        Check if two polytopes define the same set by comparing their canonical forms.
        This handles ordering and small numerical differences.
        """
        # Canonicalize both polytopes
        self_canon = cdd.matrix_copy(self.cdd_mat)
        other_canon = cdd.matrix_copy(other.cdd_mat)
        
        cdd.matrix_canonicalize(self_canon)
        cdd.matrix_canonicalize(other_canon)

        # Convert the canonicalized matrices to arrays for comparison
        self_array = np.array(self_canon.array)
        other_array = np.array(other_canon.array)

        # Sort the rows of both arrays (for consistent comparison)
        self_sorted = np.array(sorted(self_array, key=lambda row: tuple(row)))
        other_sorted = np.array(sorted(other_array, key=lambda row: tuple(row)))

        # Compare the sorted arrays with a tolerance
        return (self_sorted.shape == other_sorted.shape) and np.allclose(self_sorted, other_sorted, atol=tol)
