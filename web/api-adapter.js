/**
 * API Adapter for Flask Backend (android.py)
 *
 * This file provides a pywebview.api compatible interface that works with
 * the Flask REST API instead of the pywebview Python bridge.
 *
 * Include this BEFORE app.js when running on Android/Flask:
 * <script src="api-adapter.js"></script>
 * <script src="app.js"></script>
 */

(function() {
    'use strict';

    // API base URL - will be automatically set to current origin
    const API_BASE = window.location.origin;

    /**
     * Make API call to Flask backend
     */
    async function apiCall(endpoint, data = null) {
        const url = `${API_BASE}/api/${endpoint}`;

        const options = {
            method: data ? 'POST' : 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result;
        } catch (error) {
            console.error(`API call failed: ${endpoint}`, error);
            throw error;
        }
    }

    /**
     * Create pywebview.api compatible interface
     */
    window.pywebview = {
        api: {
            // ==================== Synthetic Data Generation ====================

            generate_simplex_points: async function(n, d, k, seed) {
                return await apiCall('generate_simplex_points', { n, d, k, seed });
            },

            save_synthetic_dataset: async function(points) {
                return await apiCall('save_synthetic_dataset', { points });
            },

            // ==================== MNIST Dataset ====================

            load_mnist: async function(max_samples, subset) {
                return await apiCall('load_mnist', { max_samples, subset });
            },

            // ==================== Upload Handling ====================

            upload_csv: async function(name, content, delimiter) {
                return await apiCall('upload_csv', { name, content, delimiter });
            },

            upload_images: async function(files) {
                return await apiCall('upload_images', { files });
            },

            list_datasets: async function() {
                return await apiCall('list_datasets');
            },

            prepare_csv_dataset: async function(dataset_id, selected_columns, handle_missing) {
                return await apiCall('prepare_csv_dataset', {
                    dataset_id,
                    selected_columns,
                    handle_missing
                });
            },

            // ==================== Embeddings ====================

            compute_embeddings: async function(dataset_id, method) {
                return await apiCall('compute_embeddings', { dataset_id, method });
            },

            // ==================== t-SNE ====================

            run_tsne: async function(dataset_id, perplexity, learning_rate, n_iter,
                                    early_exaggeration, momentum, init_method, init_data, seed) {
                return await apiCall('run_tsne', {
                    dataset_id,
                    perplexity,
                    learning_rate,
                    n_iter,
                    early_exaggeration,
                    momentum,
                    init_method,
                    init_data,
                    seed
                });
            },

            stop_tsne: async function() {
                // Note: stop functionality needs to be implemented in Flask backend
                return { success: true };
            },

            // ==================== Clustering ====================

            run_clustering: async function(dataset_id, method, k, eps, min_samples) {
                return await apiCall('run_clustering', {
                    dataset_id,
                    method,
                    k,
                    eps,
                    min_samples
                });
            },

            // ==================== Export ====================

            export_results: async function(dataset_id) {
                return await apiCall('export_results', { dataset_id });
            },

            get_image_at_index: async function(dataset_id, index) {
                return await apiCall('get_image_at_index', { dataset_id, index });
            }
        },

        // ==================== API Status ====================

        /**
         * Check if we're running in Flask/Android mode
         */
        isFlaskMode: function() {
            return true;
        },

        /**
         * Check MCP connection status
         */
        checkMCPStatus: async function() {
            try {
                const status = await apiCall('mcp_status');
                return status;
            } catch (error) {
                return { connected: false, available: false };
            }
        }
    };

    // ==================== Progress Updates ====================

    /**
     * Progress updates for t-SNE
     * In Flask mode, we use polling instead of callbacks
     */
    let progressInterval = null;

    window.startProgressPolling = function() {
        if (progressInterval) {
            clearInterval(progressInterval);
        }

        // Poll for progress updates every 500ms
        progressInterval = setInterval(async () => {
            try {
                const progress = await apiCall('tsne_progress');
                if (progress && progress.current !== undefined) {
                    window.updateProgress(progress.current, progress.total, progress.message);

                    // Stop polling when complete
                    if (progress.current >= progress.total) {
                        clearInterval(progressInterval);
                        progressInterval = null;
                    }
                }
            } catch (error) {
                // Silently fail if progress endpoint not available
            }
        }, 500);
    };

    window.stopProgressPolling = function() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    };

    // ==================== Connection Status Indicator ====================

    /**
     * Show connection status in UI
     */
    async function showConnectionStatus() {
        try {
            const health = await apiCall('health');
            const statusDiv = document.createElement('div');
            statusDiv.id = 'connection-status';
            statusDiv.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 8px 12px;
                background: ${health.mcp_connected ? '#10b981' : '#f59e0b'};
                color: white;
                border-radius: 6px;
                font-size: 0.85em;
                font-weight: 600;
                z-index: 9999;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            `;
            statusDiv.innerHTML = health.mcp_connected
                ? '🟢 MCP Connected'
                : '🟡 Local Mode';

            document.body.appendChild(statusDiv);

            // Add tooltip
            statusDiv.title = health.mcp_connected
                ? 'Connected to MCP server for heavy computations'
                : 'Using local fallback (computations may be slower)';

        } catch (error) {
            console.error('Failed to check connection status:', error);
        }
    }

    // ==================== Initialization ====================

    /**
     * Initialize Flask API adapter
     */
    function initAdapter() {
        console.log('Flask API Adapter initialized');
        console.log(`API Base URL: ${API_BASE}`);

        // Show connection status
        showConnectionStatus();

        // Dispatch ready event
        window.dispatchEvent(new Event('pywebviewready'));
    }

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAdapter);
    } else {
        initAdapter();
    }

    // ==================== Helper Functions ====================

    /**
     * Download file helper (for CSV export)
     */
    window.downloadFile = function(content, filename, type) {
        const blob = new Blob([content], { type: type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    };

})();
