// Global state
let currentDatasetId = null;
let currentResults = null;
let uploadedDatasets = [];

// Check if pywebview is available
function ensureAPI() {
    if (typeof pywebview === 'undefined' || !pywebview.api) {
        throw new Error('PyWebView API not available. Please ensure the app is running in pywebview.');
    }
}

// Wait for DOM and pywebview to be ready
function init() {
    console.log('Initializing t-SNE Explorer...');

    // Setup tab switching
    setupTabs();

    // Setup all event listeners with try-catch
    setupSyntheticDataGenerator();
    setupDataSourceManagement();
    setupTSNERunner();
    setupClustering();
    setupExport();
    setupUpload();
    setupModal();

    // Load initial data
    safeAPICall(async () => {
        await updateDataSourceDropdown();
        await refreshDatasetList();
    });
}

// Safe API call wrapper
async function safeAPICall(fn, errorMsg = 'An error occurred') {
    try {
        ensureAPI();
        return await fn();
    } catch (error) {
        console.error(errorMsg, error);
        showNotification(errorMsg + ': ' + error.message, 'error');
        return null;
    }
}

// Notification system
function showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let notif = document.getElementById('notification');
    if (!notif) {
        notif = document.createElement('div');
        notif.id = 'notification';
        document.body.appendChild(notif);
    }

    notif.textContent = message;
    notif.className = `notification ${type} show`;

    setTimeout(() => {
        notif.classList.remove('show');
    }, 4000);
}

// ==================== Tab Management ====================

function setupTabs() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;

            // Update button states
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            button.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// ==================== Synthetic Data Generation ====================

function setupSyntheticDataGenerator() {
    const generateBtn = document.getElementById('generate-btn');
    if (!generateBtn) return;

    generateBtn.addEventListener('click', async () => {
        const n = parseInt(document.getElementById('synth-n').value);
        const d = parseInt(document.getElementById('synth-d').value);
        const k = parseFloat(document.getElementById('synth-k').value);
        const seed = parseInt(document.getElementById('synth-seed').value);

        const result = await safeAPICall(
            async () => await pywebview.api.generate_simplex_points(n, d, k, seed),
            'Error generating synthetic data'
        );

        if (!result) return;

        if (!result.success) {
            showNotification(result.error, 'error');
            return;
        }

        // Display results
        const output = document.getElementById('synth-output');
        output.classList.remove('hidden');

        const stats = document.getElementById('synth-stats');
        stats.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Points</div>
                    <div class="stat-value">${result.n}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Dimensions</div>
                    <div class="stat-value">${result.d}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Target k</div>
                    <div class="stat-value">${result.k}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Actual k</div>
                    <div class="stat-value">${result.actual_k}</div>
                </div>
            </div>
            <div class="distance-info">
                <strong>Unique Distance Values:</strong> [${result.unique_distances.map(d => d.toFixed(4)).join(', ')}]
                <br>
                <strong>Range:</strong> min=${result.distances_min.toFixed(4)}, mean=${result.distances_mean.toFixed(4)}, max=${result.distances_max.toFixed(4)}
            </div>
        `;

        // Display points table
        displayPointsTable(result.points, result.d);

        // Display distance matrix
        displayDistanceMatrix(result.points);

        // Save as dataset
        const saveResult = await safeAPICall(
            async () => await pywebview.api.save_synthetic_dataset(result.points)
        );

        if (saveResult && saveResult.success) {
            currentDatasetId = saveResult.dataset_id;
            await updateDataSourceDropdown();
            await refreshDatasetList();
            showNotification('Synthetic dataset generated successfully!', 'success');
        }
    });
}

function displayPointsTable(points, d) {
    const tableContainer = document.getElementById('synth-table-container');
    const maxRows = Math.min(10, points.length);

    let tableHTML = '<div class="table-wrapper"><table class="data-table"><thead><tr><th>Point</th>';
    for (let j = 0; j < d; j++) {
        tableHTML += `<th>x<sub>${j+1}</sub></th>`;
    }
    tableHTML += '</tr></thead><tbody>';

    for (let i = 0; i < maxRows; i++) {
        tableHTML += `<tr><td>x<sub>${i+1}</sub></td>`;
        for (let j = 0; j < d; j++) {
            tableHTML += `<td>${points[i][j].toFixed(4)}</td>`;
        }
        tableHTML += '</tr>';
    }

    if (points.length > 10) {
        tableHTML += `<tr><td colspan="${d + 1}" class="more-rows">... (${points.length - 10} more rows)</td></tr>`;
    }

    tableHTML += '</tbody></table></div>';
    tableContainer.innerHTML = tableHTML;
}

function displayDistanceMatrix(points) {
    const distContainer = document.getElementById('synth-distances-container');
    const n = points.length;

    // Compute pairwise distances
    const distances = [];
    for (let i = 0; i < n; i++) {
        distances[i] = [];
        for (let j = 0; j < n; j++) {
            if (i === j) {
                distances[i][j] = 0;
            } else {
                let sum = 0;
                for (let k = 0; k < points[i].length; k++) {
                    sum += (points[i][k] - points[j][k]) ** 2;
                }
                distances[i][j] = Math.sqrt(sum);
            }
        }
    }

    // Build table HTML
    let tableHTML = '<div class="table-wrapper"><table class="data-table distance-matrix"><thead><tr><th></th>';
    for (let j = 0; j < n; j++) {
        tableHTML += `<th>x<sub>${j+1}</sub></th>`;
    }
    tableHTML += '</tr></thead><tbody>';

    for (let i = 0; i < n; i++) {
        tableHTML += `<tr><td><strong>x<sub>${i+1}</sub></strong></td>`;
        for (let j = 0; j < n; j++) {
            const cellClass = i === j ? 'diagonal' : '';
            tableHTML += `<td class="${cellClass}">${distances[i][j].toFixed(4)}</td>`;
        }
        tableHTML += '</tr>';
    }

    tableHTML += '</tbody></table></div>';
    distContainer.innerHTML = tableHTML;
}

// ==================== Data Source Management ====================

function setupDataSourceManagement() {
    const dataSource = document.getElementById('data-source');
    if (!dataSource) return;

    dataSource.addEventListener('change', async (e) => {
        const value = e.target.value;
        currentDatasetId = value === 'synthetic' ? null : value;

        // Show/hide relevant controls
        document.getElementById('csv-columns-group').style.display = 'none';
        document.getElementById('image-embed-group').style.display = 'none';
        document.getElementById('mnist-load-group').style.display = 'none';

        if (value === 'load-mnist') {
            // Show MNIST loading controls
            console.log('Showing MNIST load group');
            document.getElementById('mnist-load-group').style.display = 'block';
            currentDatasetId = null;
        } else if (value && value.startsWith('csv_')) {
            document.getElementById('csv-columns-group').style.display = 'block';
            await loadCsvColumns(value);
        } else if (value && value.startsWith('images_')) {
            document.getElementById('image-embed-group').style.display = 'block';
            currentDatasetId = value;
        } else if (value && value.startsWith('mnist_')) {
            // MNIST datasets are ready to use, no preparation needed
            currentDatasetId = value;
        }
    });

    // Prepare CSV button
    const prepareCsvBtn = document.getElementById('prepare-csv-btn');
    if (prepareCsvBtn) {
        prepareCsvBtn.addEventListener('click', async () => {
            const datasetId = document.getElementById('data-source').value;
            const checkboxes = document.querySelectorAll('#csv-columns-list input:checked');
            const selectedColumns = Array.from(checkboxes).map(cb => cb.value);
            const handleMissing = document.getElementById('csv-missing').value;

            if (selectedColumns.length === 0) {
                showNotification('Please select at least one column', 'warning');
                return;
            }

            const result = await safeAPICall(
                async () => await pywebview.api.prepare_csv_dataset(datasetId, selectedColumns, handleMissing)
            );

            if (result && result.success) {
                showNotification(`Dataset prepared: ${result.shape[0]} rows x ${result.shape[1]} columns`, 'success');
            }
        });
    }

    // MNIST loading button
    const loadMnistBtn = document.getElementById('load-mnist-btn');
    if (loadMnistBtn) {
        console.log('✓ MNIST button found, attaching click handler');
        loadMnistBtn.addEventListener('click', async () => {
            console.log('MNIST Load button clicked!');

            const subset = document.getElementById('mnist-subset').value;
            const maxSamples = parseInt(document.getElementById('mnist-samples').value);
            const statusDiv = document.getElementById('mnist-status');
            const progressContainer = document.getElementById('mnist-progress-container');
            const progressBar = document.getElementById('mnist-progress-bar');
            const progressText = document.getElementById('mnist-progress-text');

            console.log(`Loading MNIST: subset=${subset}, samples=${maxSamples}`);

            // Show progress bar
            progressContainer.style.display = 'block';
            statusDiv.style.display = 'none';
            loadMnistBtn.disabled = true;
            loadMnistBtn.textContent = 'Loading...';

            // Simulate progress steps
            const updateProgress = (percent, message) => {
                progressBar.style.width = percent + '%';
                progressText.textContent = message;
            };

            updateProgress(10, 'Connecting to OpenML...');
            await new Promise(resolve => setTimeout(resolve, 500));

            updateProgress(30, 'Downloading MNIST dataset...');

            const result = await safeAPICall(
                async () => await pywebview.api.load_mnist(maxSamples, subset),
                'Error loading MNIST dataset'
            );

            console.log('MNIST load result:', result);

            if (result && result.success) {
                updateProgress(70, 'Processing images...');
                await new Promise(resolve => setTimeout(resolve, 300));

                updateProgress(90, 'Creating dataset...');
                await new Promise(resolve => setTimeout(resolve, 300));

                updateProgress(100, 'Complete!');
                await new Promise(resolve => setTimeout(resolve, 500));

                // Hide progress, show success message
                progressContainer.style.display = 'none';
                statusDiv.style.display = 'block';
                statusDiv.textContent = `✓ ${result.message}`;
                statusDiv.style.color = '#10b981';
                statusDiv.style.background = '#d1fae5';

                showNotification(result.message, 'success');
                await updateDataSourceDropdown();
                await refreshDatasetList();

                // Auto-select the newly loaded dataset
                const datasets = await safeAPICall(async () => await pywebview.api.list_datasets());
                if (datasets && datasets.length > 0) {
                    const mnistDataset = datasets.find(d => d.type === 'mnist');
                    if (mnistDataset) {
                        dataSource.value = mnistDataset.id;
                        currentDatasetId = mnistDataset.id;
                        document.getElementById('mnist-load-group').style.display = 'none';
                    }
                }
            } else {
                progressContainer.style.display = 'none';
                statusDiv.style.display = 'block';
                statusDiv.textContent = `✗ Failed to load MNIST: ${result?.error || 'Unknown error'}`;
                statusDiv.style.color = '#ef4444';
                statusDiv.style.background = '#fee2e2';
            }

            loadMnistBtn.disabled = false;
            loadMnistBtn.textContent = 'Load MNIST Dataset';
        });
    } else {
        console.error('✗ MNIST button NOT found!');
    }

    // Compute embeddings button
    const computeEmbedBtn = document.getElementById('compute-embed-btn');
    if (computeEmbedBtn) {
        computeEmbedBtn.addEventListener('click', async () => {
            const datasetId = document.getElementById('data-source').value;
            const method = document.getElementById('embed-method').value;
            const statusDiv = document.getElementById('embed-status');

            statusDiv.textContent = 'Computing embeddings...';
            statusDiv.className = 'embed-status computing';

            const result = await safeAPICall(
                async () => await pywebview.api.compute_embeddings(datasetId, method)
            );

            if (result && result.success) {
                statusDiv.textContent = `✓ Embeddings computed using ${result.method}: ${result.shape[0]}x${result.shape[1]}`;
                statusDiv.className = 'embed-status success';
                showNotification('Embeddings computed successfully!', 'success');
            } else {
                statusDiv.textContent = '✗ Failed to compute embeddings';
                statusDiv.className = 'embed-status error';
            }
        });
    }
}

async function updateDataSourceDropdown() {
    const select = document.getElementById('data-source');
    if (!select) return;

    const datasets = await safeAPICall(async () => await pywebview.api.list_datasets());
    if (!datasets) return;

    // Clear existing options except first three (includes Load MNIST Dataset)
    while (select.options.length > 3) {
        select.remove(3);
    }

    // Add dataset options
    datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.id;

        if (dataset.type === 'csv') {
            option.textContent = `📊 CSV: ${dataset.name} (${dataset.shape[0]}×${dataset.shape[1]})`;
        } else if (dataset.type === 'images') {
            option.textContent = `🖼️ Images: ${dataset.count} files`;
        } else if (dataset.type === 'synthetic') {
            option.textContent = `🔢 Synthetic: ${dataset.shape[0]}×${dataset.shape[1]}`;
        } else if (dataset.type === 'mnist') {
            option.textContent = `✏️ ${dataset.name}`;
        }

        select.appendChild(option);

        if (dataset.id === currentDatasetId) {
            select.value = dataset.id;
        }
    });
}

async function loadCsvColumns(datasetId) {
    // This would need a separate API call to get column info
    // For now, it's a placeholder
}

// ==================== t-SNE Runner ====================

function setupTSNERunner() {
    const runBtn = document.getElementById('run-tsne-btn');
    const stopBtn = document.getElementById('stop-tsne-btn');
    const initMethodSelect = document.getElementById('init-method');
    const customInitGroup = document.getElementById('custom-init-group');

    // Handle initialization method change
    if (initMethodSelect && customInitGroup) {
        initMethodSelect.addEventListener('change', (e) => {
            if (e.target.value === 'custom') {
                customInitGroup.style.display = 'block';
            } else {
                customInitGroup.style.display = 'none';
            }
        });
    }

    if (runBtn) {
        runBtn.addEventListener('click', async () => {
            const datasetId = document.getElementById('data-source').value;

            if (!datasetId) {
                showNotification('Please select a data source first', 'warning');
                return;
            }

            const params = {
                perplexity: parseInt(document.getElementById('perplexity').value),
                learning_rate: parseInt(document.getElementById('learning-rate').value),
                n_iter: parseInt(document.getElementById('iterations').value),
                early_exaggeration: parseInt(document.getElementById('early-exag').value),
                momentum: parseFloat(document.getElementById('momentum').value),
                init_method: document.getElementById('init-method').value,
                seed: parseInt(document.getElementById('tsne-seed').value)
            };

            // Handle custom initialization
            let init_data = null;
            if (params.init_method === 'custom') {
                const customInitText = document.getElementById('custom-init-coords').value.trim();
                if (customInitText) {
                    try {
                        init_data = JSON.parse(customInitText);
                    } catch (e) {
                        showNotification('Invalid JSON format for custom initialization', 'error');
                        return;
                    }
                } else {
                    showNotification('Please provide custom initialization coordinates', 'warning');
                    return;
                }
            }

            // Show progress
            const progressContainer = document.getElementById('progress-container');
            progressContainer.classList.remove('hidden');
            runBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';

            const result = await safeAPICall(
                async () => await pywebview.api.run_tsne(
                    datasetId,
                    params.perplexity,
                    params.learning_rate,
                    params.n_iter,
                    params.early_exaggeration,
                    params.momentum,
                    params.init_method,
                    init_data,
                    params.seed
                ),
                'Error running t-SNE'
            );

            progressContainer.classList.add('hidden');
            runBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';

            if (result && result.success) {
                currentResults = result;
                displayResults(result, datasetId);

                // Show clustering section for image/MNIST datasets
                const clusteringSection = document.getElementById('clustering-section');
                const isImageDataset = datasetId && (datasetId.startsWith('images_') || datasetId.startsWith('mnist_'));

                if (clusteringSection) {
                    if (isImageDataset) {
                        clusteringSection.style.display = 'block';
                    } else {
                        clusteringSection.style.display = 'none';
                    }
                }

                showNotification('t-SNE completed successfully!', 'success');
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            await safeAPICall(async () => await pywebview.api.stop_tsne());
            showNotification('t-SNE stopped', 'info');
        });
    }
}

// Progress callback
window.updateProgress = (current, total, message) => {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    if (progressBar && progressText) {
        const percentage = (current / total) * 100;
        progressBar.style.width = percentage + '%';
        progressText.textContent = message;
    }
};

// ==================== Results Display ====================

function displayResults(result, datasetId) {
    console.log('Displaying results:', {
        Y_shape: [result.Y.length, result.Y[0]?.length],
        P_shape: [result.P?.length, result.P?.[0]?.length],
        Q_shape: [result.Q?.length, result.Q?.[0]?.length],
        C_history_length: result.C_history?.length,
        has_labels: result.has_labels
    });

    document.getElementById('results-section').style.display = 'block';

    const Y = result.Y;

    // 2D Scatter Plot
    plotScatter(Y, datasetId, result.labels);

    // Cost Plot
    if (result.C_history && result.C_history.length > 0) {
        plotCost(result.C_history);
    } else {
        console.warn('No cost history available');
    }

    // Matrix Heatmaps and Grids
    if (result.P && result.P.length > 0) {
        plotMatrix(result.P, 'p-matrix-plot', 'P Matrix (High-D Affinities)');
        displayMatrixGrid(result.P, 'p-matrix-grid', 'P', 'y');
    } else {
        console.warn('P matrix not available');
    }

    if (result.Q && result.Q.length > 0) {
        plotMatrix(result.Q, 'q-matrix-plot', 'Q Matrix (Low-D Affinities)');
        displayMatrixGrid(result.Q, 'q-matrix-grid', 'Q', 'y');
    } else {
        console.warn('Q matrix not available');
    }

    // Distances between y_i in the embedding
    const distances = computePairwiseDistances(Y);
    result.D = distances;
    plotMatrix(distances, 'd-matrix-plot', 'Distances Between y_i (Embedding)');
    displayMatrixGrid(distances, 'd-matrix-grid', 'D', 'y');

    // Coordinates Table
    displayCoordinatesTable(Y);
}

function computePairwiseDistances(Y) {
    const n = Y.length;
    const distances = new Array(n);

    for (let i = 0; i < n; i++) {
        distances[i] = new Array(n);
        for (let j = 0; j < n; j++) {
            if (i === j) {
                distances[i][j] = 0;
                continue;
            }
            const dx = Y[i][0] - Y[j][0];
            const dy = Y[i][1] - Y[j][1];
            distances[i][j] = Math.sqrt(dx * dx + dy * dy);
        }
    }

    return distances;
}

function plotScatter(Y, datasetId, labels) {
    // Color palette for MNIST digits (0-9)
    const digitColors = [
        '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#95a5a6', '#34495e', '#c0392b'
    ];

    let trace;

    if (labels && labels.length === Y.length) {
        // Create separate trace for each digit class
        const traces = [];
        const uniqueLabels = [...new Set(labels)].sort((a, b) => a - b);

        uniqueLabels.forEach(label => {
            const indices = labels.map((l, i) => l === label ? i : -1).filter(i => i >= 0);
            const color = digitColors[label % digitColors.length];

            traces.push({
                x: indices.map(i => Y[i][0]),
                y: indices.map(i => Y[i][1]),
                mode: 'markers',
                type: 'scatter',
                name: `Digit ${label}`,
                marker: {
                    size: 8,
                    color: color,
                    line: {
                        color: '#ffffff',
                        width: 1
                    }
                },
                hovertext: indices.map(i => `Digit ${label}<br>Point ${i+1}<br>Dim 1: ${Y[i][0].toFixed(3)}<br>Dim 2: ${Y[i][1].toFixed(3)}`),
                hoverinfo: 'text'
            });
        });

        const layout = {
            title: {
                text: 't-SNE Embedding (Colored by True Labels)',
                font: { size: 18, family: 'Segoe UI, sans-serif' }
            },
            xaxis: { title: 'Dimension 1', gridcolor: '#e0e0e0' },
            yaxis: { title: 'Dimension 2', gridcolor: '#e0e0e0' },
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: '#ffffff',
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.2
            }
        };

        Plotly.newPlot('tsne-plot', traces, layout);
    } else {
        // Default plot without labels
        trace = {
            x: Y.map(p => p[0]),
            y: Y.map(p => p[1]),
            mode: 'markers+text',
            type: 'scatter',
            marker: {
                size: 10,
                color: '#667eea',
                line: {
                    color: '#ffffff',
                    width: 1
                }
            },
            text: Y.map((p, i) => `y${i+1}`),
            textposition: 'top center',
            textfont: {
                size: 10,
                color: '#1f2937'
            },
            hovertext: Y.map((p, i) => `Point y${i+1}<br>Dim 1: ${p[0].toFixed(3)}<br>Dim 2: ${p[1].toFixed(3)}`),
            hoverinfo: 'text'
        };

        const layout = {
            title: {
                text: 't-SNE Embedding',
                font: { size: 18, family: 'Segoe UI, sans-serif' }
            },
            xaxis: { title: 'Dimension 1', gridcolor: '#e0e0e0' },
            yaxis: { title: 'Dimension 2', gridcolor: '#e0e0e0' },
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: '#ffffff'
        };

        Plotly.newPlot('tsne-plot', [trace], layout);
    }

    // Add click handler for images
    if (datasetId && (datasetId.startsWith('images_') || datasetId.startsWith('mnist_'))) {
        document.getElementById('tsne-plot').on('plotly_click', async (data) => {
            const pointIndex = data.points[0].pointIndex;
            await showImagePreview(datasetId, pointIndex);
        });
    }
}

function plotCost(costHistory) {
    const trace = {
        y: costHistory,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#e74c3c', width: 2 }
    };

    const layout = {
        title: {
            text: 'KL Divergence over Iterations',
            font: { size: 18, family: 'Segoe UI, sans-serif' }
        },
        xaxis: { title: 'Iteration', gridcolor: '#e0e0e0' },
        yaxis: { title: 'Cost (KL Divergence)', gridcolor: '#e0e0e0' },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot('cost-plot', [trace], layout);
}

function plotMatrix(matrix, elementId, title) {
    const maxSize = 100;
    const n = matrix.length;

    let displayMatrix = matrix;
    if (n > maxSize) {
        const step = Math.ceil(n / maxSize);
        displayMatrix = [];
        for (let i = 0; i < n; i += step) {
            const row = [];
            for (let j = 0; j < n; j += step) {
                row.push(matrix[i][j]);
            }
            displayMatrix.push(row);
        }
    }

    const trace = {
        z: displayMatrix,
        type: 'heatmap',
        colorscale: 'Viridis'
    };

    const layout = {
        title: {
            text: title + (n > maxSize ? ' (downsampled)' : ''),
            font: { size: 16, family: 'Segoe UI, sans-serif' }
        },
        xaxis: { title: 'Point j' },
        yaxis: { title: 'Point i' },
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot(elementId, [trace], layout);
}

function displayCoordinatesTable(Y) {
    const coordsTable = document.getElementById('coords-table');
    let html = '<div class="table-wrapper"><table class="data-table"><thead><tr><th>Point</th><th>Dim 1</th><th>Dim 2</th></tr></thead><tbody>';

    const maxRows = Math.min(20, Y.length);
    for (let i = 0; i < maxRows; i++) {
        html += `<tr><td>y${i+1}</td><td>${Y[i][0].toFixed(4)}</td><td>${Y[i][1].toFixed(4)}</td></tr>`;
    }

    if (Y.length > 20) {
        html += `<tr><td colspan="3" class="more-rows">... (${Y.length - 20} more rows)</td></tr>`;
    }

    html += '</tbody></table></div>';
    coordsTable.innerHTML = html;
}

function displayMatrixGrid(matrix, elementId, matrixName, labelPrefix = '') {
    const gridContainer = document.getElementById(elementId);
    const n = matrix.length;
    const maxDisplay = 20; // Show max 20x20 for performance

    let html = '<div class="table-wrapper" style="max-height: 500px; overflow: auto;"><table class="data-table matrix-grid"><thead><tr><th></th>';

    // Column headers
    const displayN = Math.min(n, maxDisplay);
    for (let j = 0; j < displayN; j++) {
        const label = labelPrefix ? `${labelPrefix}${j + 1}` : `${j + 1}`;
        html += `<th>${label}</th>`;
    }
    if (n > maxDisplay) {
        html += '<th>...</th>';
    }
    html += '</tr></thead><tbody>';

    // Matrix rows
    for (let i = 0; i < displayN; i++) {
        const label = labelPrefix ? `${labelPrefix}${i + 1}` : `${i + 1}`;
        html += `<tr><td><strong>${label}</strong></td>`;
        for (let j = 0; j < displayN; j++) {
            const value = matrix[i][j];
            const cellClass = i === j ? 'diagonal' : '';
            html += `<td class="${cellClass}">${value.toFixed(6)}</td>`;        
        }
        if (n > maxDisplay) {
            html += '<td>...</td>';
        }
        html += '</tr>';
    }

    if (n > maxDisplay) {
        html += `<tr><td><strong>...</strong></td>${'<td>...</td>'.repeat(displayN + 1)}</tr>`;
    }

    html += '</tbody></table></div>';
    html += `<p class="info">Showing ${displayN}x${displayN} of ${n}x${n} matrix</p>`;

    gridContainer.innerHTML = html;
}

function toggleMatrixView(matrixName, viewType) {
    const plotId = `${matrixName.toLowerCase()}-matrix-plot`;
    const gridId = `${matrixName.toLowerCase()}-matrix-grid`;

    const plotDiv = document.getElementById(plotId);
    const gridDiv = document.getElementById(gridId);

    if (viewType === 'heatmap') {
        plotDiv.style.display = 'block';
        gridDiv.style.display = 'none';
    } else if (viewType === 'grid') {
        plotDiv.style.display = 'none';
        gridDiv.style.display = 'block';
    }
}

// ==================== Clustering ====================

// Auto-clustering removed - user can manually run clustering from the Clustering section
// async function runAutoClusteringForMNIST(datasetId) {
//     ...
// }

function setupClustering() {
    const methodSelect = document.getElementById('cluster-method');
    const runBtn = document.getElementById('run-cluster-btn');

    if (methodSelect) {
        methodSelect.addEventListener('change', (e) => {
            const method = e.target.value;
            document.getElementById('kmeans-params').classList.toggle('hidden', method !== 'kmeans');
            document.getElementById('dbscan-params').classList.toggle('hidden', method !== 'dbscan');
        });
    }

    if (runBtn) {
        runBtn.addEventListener('click', async () => {
            const datasetId = document.getElementById('data-source').value;
            const method = document.getElementById('cluster-method').value;

            const params = {
                k: parseInt(document.getElementById('kmeans-k').value) || 3,
                eps: parseFloat(document.getElementById('dbscan-eps').value) || 0.5,
                min_samples: parseInt(document.getElementById('dbscan-minsamples').value) || 5
            };

            const result = await safeAPICall(
                async () => await pywebview.api.run_clustering(
                    datasetId, method, params.k, params.eps, params.min_samples
                )
            );

            if (result && result.success) {
                updateScatterWithClusters(result.labels, result.summary);
                showNotification('Clustering completed!', 'success');
            }
        });
    }
}

function updateScatterWithClusters(labels, summary) {
    const Y = currentResults.Y;

    const trace = {
        x: Y.map(p => p[0]),
        y: Y.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: 10,
            color: labels,
            colorscale: 'Viridis',
            showscale: true,
            line: { color: '#ffffff', width: 1 }
        },
        text: Y.map((p, i) => `Point y${i + 1}<br>Cluster: ${labels[i]}<br>Dim 1: ${p[0].toFixed(3)}<br>Dim 2: ${p[1].toFixed(3)}`),
        hoverinfo: 'text'
    };

    const layout = {
        title: {
            text: 't-SNE Embedding (Colored by Cluster)',
            font: { size: 18, family: 'Segoe UI, sans-serif' }
        },
        xaxis: { title: 'Dimension 1', gridcolor: '#e0e0e0' },
        yaxis: { title: 'Dimension 2', gridcolor: '#e0e0e0' },
        hovermode: 'closest',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot('tsne-plot', [trace], layout);

    // Display summary
    displayClusterSummary(summary);
}

function displayClusterSummary(summary) {
    const summaryDiv = document.getElementById('cluster-summary');
    let html = '<h4>Cluster Summary</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>Cluster</th><th>Count</th></tr></thead><tbody>';

    summary.forEach(item => {
        html += `<tr><td>${item.label}</td><td>${item.count}</td></tr>`;
    });

    html += '</tbody></table></div>';
    summaryDiv.innerHTML = html;
}

// ==================== Export ====================

function setupExport() {
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', async () => {
            const datasetId = document.getElementById('data-source').value;
            const result = await safeAPICall(
                async () => await pywebview.api.export_results(datasetId)
            );

            if (result && result.success) {
                downloadFile(result.csv, 'tsne_results.csv', 'text/csv');
                showNotification('Results exported successfully!', 'success');
            }
        });
    }
}

window.downloadMatrix = function(matrixType) {
    if (!currentResults) {
        showNotification('No results to export', 'warning');
        return;
    }

    let matrix = null;
    if (matrixType === 'P') matrix = currentResults.P;
    if (matrixType === 'Q') matrix = currentResults.Q;
    if (matrixType === 'D') matrix = currentResults.D;

    if (!matrix) {
        showNotification(`Matrix ${matrixType} not available`, 'warning');
        return;
    }

    const csv = matrix.map(row => row.join(',')).join('\n');
    const filename = matrixType === 'D' ? 'embedding_distances.csv' : `${matrixType}_matrix.csv`;
    downloadFile(csv, filename, 'text/csv');
    showNotification(`${matrixType} matrix exported!`, 'success');
};

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type: type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ==================== Upload ====================

function setupUpload() {
    setupCSVUpload();
    setupImageUpload();
}

function setupCSVUpload() {
    const uploadBtn = document.getElementById('csv-upload-btn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('csv-upload');
            const files = fileInput.files;

            if (files.length === 0) {
                showNotification('Please select CSV file(s)', 'warning');
                return;
            }

            for (let file of files) {
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const content = e.target.result;
                    const result = await safeAPICall(
                        async () => await pywebview.api.upload_csv(file.name, content, ',')
                    );

                    if (result && result.success) {
                        showNotification(`Uploaded ${file.name}`, 'success');
                        await updateDataSourceDropdown();
                        await refreshDatasetList();

                        if (result.numeric_columns.length > 0) {
                            displayCSVColumns(result.numeric_columns);
                        }
                    }
                };
                reader.readAsText(file);
            }
        });
    }
}

function displayCSVColumns(columns) {
    const columnsList = document.getElementById('csv-columns-list');
    if (!columnsList) return;

    columnsList.innerHTML = '';
    columns.forEach(col => {
        const label = document.createElement('label');
        label.className = 'checkbox-label';
        label.innerHTML = `<input type="checkbox" value="${col}" checked> ${col}`;
        columnsList.appendChild(label);
    });
}

function setupImageUpload() {
    const imageUploadBtn = document.getElementById('image-upload-btn');
    const folderUploadBtn = document.getElementById('folder-upload-btn');
    const imageInput = document.getElementById('image-upload');
    const folderInput = document.getElementById('folder-upload');

    if (imageUploadBtn) {
        imageUploadBtn.addEventListener('click', () => imageInput.click());
    }

    if (folderUploadBtn) {
        folderUploadBtn.addEventListener('click', () => folderInput.click());
    }

    if (imageInput) {
        imageInput.addEventListener('change', (e) => handleImageUpload(e.target.files));
    }

    if (folderInput) {
        folderInput.addEventListener('change', (e) => handleImageUpload(e.target.files));
    }
}

async function handleImageUpload(files) {
    if (files.length === 0) return;

    showNotification('Uploading images...', 'info');
    const imageFiles = [];

    for (let file of files) {
        if (!file.type.startsWith('image/')) continue;

        const content = await readFileAsDataURL(file);
        imageFiles.push({ name: file.name, content: content });
    }

    if (imageFiles.length === 0) {
        showNotification('No valid image files found', 'warning');
        return;
    }

    const result = await safeAPICall(
        async () => await pywebview.api.upload_images(imageFiles)
    );

    if (result && result.success) {
        showNotification(`Uploaded ${result.count} images`, 'success');
        await updateDataSourceDropdown();
        await refreshDatasetList();
    }
}

function readFileAsDataURL(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.readAsDataURL(file);
    });
}

async function refreshDatasetList() {
    const datasets = await safeAPICall(async () => await pywebview.api.list_datasets());
    if (!datasets) return;

    const container = document.getElementById('datasets-container');
    if (!container) return;

    if (datasets.length === 0) {
        container.innerHTML = '<p class="empty-state">📭 No datasets uploaded yet</p>';
        return;
    }

    let html = '<ul class="dataset-list">';
    datasets.forEach(dataset => {
        let icon = '📊';
        let label = '';

        if (dataset.type === 'csv') {
            icon = '📊';
            label = `CSV: ${dataset.name} (${dataset.shape[0]}×${dataset.shape[1]})`;
        } else if (dataset.type === 'images') {
            icon = '🖼️';
            label = `Images: ${dataset.count} files`;
        } else if (dataset.type === 'synthetic') {
            icon = '🔢';
            label = `Synthetic: ${dataset.shape[0]}×${dataset.shape[1]}`;
        } else if (dataset.type === 'mnist') {
            icon = '✏️';
            label = `${dataset.name}`;
        }

        html += `<li class="dataset-item"><span class="dataset-icon">${icon}</span><span class="dataset-label">${label}</span></li>`;
    });
    html += '</ul>';

    container.innerHTML = html;
}

// ==================== Modal ====================

function setupModal() {
    const closeBtn = document.querySelector('.modal-close');
    const modal = document.getElementById('image-modal');

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
    }

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
}

async function showImagePreview(datasetId, index) {
    const result = await safeAPICall(
        async () => await pywebview.api.get_image_at_index(datasetId, index)
    );

    if (result && result.success) {
        const modal = document.getElementById('image-modal');
        const modalTitle = document.getElementById('modal-title');
        const modalImage = document.getElementById('modal-image');

        modalTitle.textContent = result.name;
        modalImage.src = result.image;
        modal.classList.remove('hidden');
    }
}

// ==================== Initialize ====================

// Wait for pywebview API to be available
function waitForPyWebView() {
    return new Promise((resolve) => {
        if (typeof pywebview !== 'undefined' && pywebview.api) {
            console.log('PyWebView API already available');
            resolve();
        } else {
            console.log('Waiting for PyWebView API...');
            window.addEventListener('pywebviewready', () => {
                console.log('PyWebView API ready!');
                resolve();
            });

            // Fallback: poll for API availability
            const checkInterval = setInterval(() => {
                if (typeof pywebview !== 'undefined' && pywebview.api) {
                    console.log('PyWebView API detected via polling');
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);

            // Timeout after 10 seconds
            setTimeout(() => {
                clearInterval(checkInterval);
                if (typeof pywebview === 'undefined' || !pywebview.api) {
                    console.error('PyWebView API failed to load within 10 seconds');
                    showNotification('Failed to connect to backend. Please restart the application.', 'error');
                }
            }, 10000);
        }
    });
}

// Initialize when both DOM and pywebview are ready
async function startApp() {
    console.log('Starting app initialization...');

    // Wait for pywebview API
    await waitForPyWebView();

    // Initialize the app
    init();

    console.log('App initialization complete!');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startApp);
} else {
    startApp();
}
