def save_curvatures(times, kappas, filename="curvature.pkl"):
    """Save curvatures in a pickle."""
    pickle.dump([times, kappas], open(filename, "wb"))


def load_curvature(filename="curvature"):
    """Load curvatures from a pickle."""
    times, kappas = pickle.load(open(filename, "rb"))
    return times, kappas

def _compute_distance_geodesic(G):
    """Geodesic distance matrix."""
    return scg.floyd_warshall(
        nx.adjacency_matrix(G, weight="weight"), directed=True, unweighted=False
    )

def _edge_curvature(
    edge,
    measures,
    geodesic_distances,
    measure_cutoff=1e-6,
    sinkhorn_regularisation=0,
    weighted_curvature=False,
):
    """Compute curvature for an edge."""
    node_x, node_y = edge
    m_x, m_y = measures[node_x], measures[node_y]

    Nx = np.where(m_x >= measure_cutoff * np.max(m_x))[0]
    Ny = np.where(m_y >= measure_cutoff * np.max(m_y))[0]

    m_x, m_y = m_x[Nx], m_y[Ny]
    m_x /= m_x.sum()
    m_y /= m_y.sum()

    distances_xy = geodesic_distances[np.ix_(Nx, Ny)]

    if sinkhorn_regularisation > 0:
        wasserstein_distance = ot.sinkhorn2(m_x, m_y, distances_xy, sinkhorn_regularisation)[0]
    else:
        wasserstein_distance = ot.emd2(m_x, m_y, distances_xy)

    if weighted_curvature:
        return geodesic_distances[node_x, node_y] - wasserstein_distance
    return 1.0 - wasserstein_distance / geodesic_distances[node_x, node_y]

def _construct_laplacian(graph, use_spectral_gap=True):
    """Laplacian matrix."""
    degrees = np.array([graph.degree(i, weight="weight") for i in graph.nodes])
    laplacian = nx.laplacian_matrix(graph).dot(sc.sparse.diags(1.0 / degrees))
    if use_spectral_gap and len(graph) > 3:
        spectral_gap = sorted(abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0]))[1]
        laplacian /= spectral_gap
    return laplacian

def _heat_kernel(measure, laplacian, timestep):
    """Compute matrix exponential on a measure."""
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, measure)

def compute_curvatures(
    graph,
    times,
    n_workers=1,
    use_spectral_gap=True,
    measure_cutoff=1e-6,
    sinkhorn_regularisation=0,
    weighted_curvature=False,
    filename="curvature.pkl",
):
    """Computes the curvatures of edges.

    Args:
        graph (networkx graph): graph to consider
        times (array): array of times to compute curvature
        n_workers (int): number of workers for multiprocessing
        use_spectral_gap (bool): to normalise time by the spectral gap of laplacian
        measure_cutoff (float): cutoff of the measures, in [0, 1], with no cutoff at 0
        sinkhorn_regularisation (float): Sinkhorn regularisation, when 0, no sinkhorn is applied
        weighted_curvature (bool): if True, the curvature is multiplied by the original edge weight
        filename (str): pickle filename to save curvatures at each time step
    """
    # Check for self-loops
    if nx.number_of_selfloops(graph) > 0:
        raise Exception("A graph with self-loops will not work!")

    # Check for connectedness
    degrees = [graph.degree(n) for n in graph.nodes]
    assert ~(np.array(degrees) == 0).any(), "Graph is not connected!"

    laplacian = _construct_laplacian(graph, use_spectral_gap)
    geodesic_distances = _compute_distance_geodesic(graph)

    times_with_zero = np.insert(times, 0, 0.0)

    kappas = np.ones([len(times), len(graph.edges())])
    measures = list(np.eye(len(graph)))
    display_all_positive = False
    with multiprocessing.Pool(n_workers) as pool:
        chunksize = max(1, int(len(graph.edges) / n_workers))
        for time_index in tqdm(range(len(times))):
        
            measures = pool.map(
                partial(
                    _heat_kernel,
                    laplacian=laplacian,
                    timestep=times_with_zero[time_index + 1] - times_with_zero[time_index],
                ),
                measures,
                chunksize=chunksize,
            )

            kappas[time_index] = pool.map(
                partial(
                    _edge_curvature,
                    measures=measures,
                    geodesic_distances=geodesic_distances,
                    measure_cutoff=measure_cutoff,
                    sinkhorn_regularisation=sinkhorn_regularisation,
                    weighted_curvature=weighted_curvature,
                ),
                graph.edges,
                chunksize=chunksize,
            )

            if all(kappas[time_index] > 0) and display_all_positive:
                display_all_positive = False

            save_curvatures(times[:time_index], kappas[:time_index], filename=filename)

    return kappas