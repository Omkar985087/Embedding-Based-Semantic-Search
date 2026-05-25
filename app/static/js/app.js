/**
 * ==========================================================================
 * EMBEDDING-BASED SEMANTIC SEARCH FRONTEND CONTROLLER
 * ==========================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    // Current application state
    const state = {
        activeTab: "search-playground",
        searchFilter: "",
        visualizerQuery: "how to page on-call",
        visualizerNodes: [],
        visualizerQueryCoord: null,
        visualizerNearestNeighbors: [],
        hoveredNode: null,
        benchmarkData: null,
        chartInstance: null,
        ragTypingTimer: null
    };

    // DOM Element References
    const elements = {
        // Nav tabs
        navButtons: document.querySelectorAll(".nav-btn"),
        tabContents: document.querySelectorAll(".tab-content"),
        engineModeText: document.getElementById("engine-mode-text"),
        faissText: document.getElementById("faiss-text"),
        engineModeBadge: document.getElementById("engine-mode-badge"),

        // Search Playground
        queryInput: document.getElementById("query-input"),
        searchBtn: document.getElementById("search-btn"),
        filterBtns: document.querySelectorAll(".filter-btn"),
        tfidfResultsList: document.getElementById("tfidf-results-list"),
        vectorResultsList: document.getElementById("vector-results-list"),

        // RAG Pipeline
        ragQueryInput: document.getElementById("rag-query-input"),
        ragSubmitBtn: document.getElementById("rag-submit-btn"),
        ragStepTabs: document.querySelectorAll(".step-tab"),
        ragStepSlides: document.querySelectorAll(".step-slide"),
        ragRetrievedChunks: document.getElementById("rag-retrieved-chunks"),
        ragPromptCode: document.getElementById("rag-prompt-code"),
        ragResponseText: document.getElementById("rag-response-text"),
        ragTyping: document.getElementById("rag-typing"),

        // Vector Visualizer
        visQueryInput: document.getElementById("vis-query-input"),
        visQueryBtn: document.getElementById("vis-query-btn"),
        vectorCanvas: document.getElementById("vector-canvas"),
        visNodeDetails: document.getElementById("vis-node-details"),
        resetCanvasBtn: document.getElementById("reset-canvas-btn"),

        // Relevance Simulator
        runBenchmarkBtn: document.getElementById("run-benchmark-btn"),
        runBenchmarkBtnSub: document.getElementById("run-benchmark-btn-sub"),
        benchmarkMetricsGrid: document.getElementById("benchmark-metrics-grid"),
        benchmarkResultsContainer: document.getElementById("benchmark-results-container"),
        benchmarkPreRun: document.getElementById("benchmark-pre-run"),
        gainVal: document.getElementById("gain-val"),
        vectorMrrVal: document.getElementById("vector-mrr-val"),
        tfidfMrrVal: document.getElementById("tfidf-mrr-val"),
        metricsChart: document.getElementById("metrics-chart"),
        scenariosContainer: document.getElementById("scenarios-container")
    };

    /* ==========================================================================
       TAB MANAGEMENT
       ========================================================================== */
    elements.navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            
            // Toggle active nav button
            elements.navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Toggle active tab content
            elements.tabContents.forEach(c => c.classList.remove("active"));
            const targetContent = document.getElementById(tabId);
            if (targetContent) targetContent.classList.add("active");

            state.activeTab = tabId;

            // Trigger visualizations if transitioning to map or relevance tabs
            if (tabId === "vector-visualizer") {
                initCanvas();
                loadVisualizerData(true);
            }
        });
    });

    /* ==========================================================================
       API UTILS
       ========================================================================== */
    async function apiFetch(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error("Fetch failure:", error);
            showNotification(`Failed to connect to backend: ${error.message}`, "danger");
            return null;
        }
    }

    function showNotification(msg, type = "info") {
        console.log(`[Notification - ${type}]: ${msg}`);
    }

    // Set connection status badge
    async function updateSystemState() {
        // Perform a quick visualize fetch to check what engine we are running
        const data = await apiFetch("/api/visualize");
        if (data) {
            elements.engineModeText.textContent = data.engine_mode;
            
            if (data.engine_mode.includes("all-MiniLM")) {
                elements.engineModeBadge.style.border = "1px solid rgba(139, 92, 246, 0.4)";
                elements.faissText.textContent = "FlatIP GPU/CPU Enabled";
                elements.faissText.classList.add("text-active");
                elements.faissText.classList.remove("text-dim");
            } else {
                elements.engineModeBadge.style.border = "1px solid rgba(234, 179, 8, 0.3)";
                elements.faissText.textContent = "Offline Numpy Flat Engine";
                elements.faissText.classList.remove("text-active");
                elements.faissText.classList.add("text-dim");
            }
        }
    }

    /* ==========================================================================
       TAB 1: SEARCH PLAYGROUND
       ========================================================================== */
    elements.searchBtn.addEventListener("click", runSearchPlayground);
    elements.queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") runSearchPlayground();
    });

    // Handle source filter buttons
    elements.filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            elements.filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.searchFilter = btn.getAttribute("data-source");
            runSearchPlayground();
        });
    });

    async function runSearchPlayground() {
        const query = elements.queryInput.value.trim();
        if (!query) return;

        // Set loader state
        elements.tfidfResultsList.innerHTML = `<div class="empty-state">Searching lexical catalog...</div>`;
        elements.vectorResultsList.innerHTML = `<div class="empty-state">Searching vector space embeddings...</div>`;

        let url = `/api/search?q=${encodeURIComponent(query)}`;
        if (state.searchFilter) {
            url += `&source=${encodeURIComponent(state.searchFilter)}`;
        }

        const data = await apiFetch(url);
        if (!data) return;

        // Update system state status badge dynamically based on model
        if (data.engine_mode) {
            elements.engineModeText.textContent = data.engine_mode;
        }

        // Render TF-IDF Results
        renderTFIDFResults(data.tfidf_results, query);

        // Render Vector Similarity Results
        renderVectorResults(data.vector_results, query);
    }

    function renderTFIDFResults(results, query) {
        if (!results || results.length === 0) {
            elements.tfidfResultsList.innerHTML = `<div class="empty-state">No matching words found. Score: 0.00</div>`;
            return;
        }

        const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
        
        elements.tfidfResultsList.innerHTML = results.map((doc, idx) => {
            // Highlight actual keyword overlaps
            let highlightedContent = doc.content;
            let hasOverlap = false;
            
            queryWords.forEach(word => {
                const regex = new RegExp(`\\b(${word})\\b`, "gi");
                if (regex.test(highlightedContent)) {
                    highlightedContent = highlightedContent.replace(regex, `<span class="match-hl">$1</span>`);
                    hasOverlap = true;
                }
            });

            const scorePercent = Math.round(doc.score * 100);
            
            return `
                <div class="result-card rank-${idx} ${doc.score > 0 ? 'active' : ''}">
                    <div class="card-top">
                        <span class="card-source ${doc.source}">✦ ${doc.source}</span>
                        <span class="card-score">Score: ${doc.score.toFixed(3)}</span>
                    </div>
                    <h4 class="card-title">${doc.title}</h4>
                    <p class="card-content">${highlightedContent}</p>
                    <div class="score-visual-container">
                        <div class="score-visual-bar" style="width: ${scorePercent}%"></div>
                    </div>
                </div>
            `;
        }).join("");
    }

    function renderVectorResults(results, query) {
        if (!results || results.length === 0) {
            elements.vectorResultsList.innerHTML = `<div class="empty-state">No semantic matching files.</div>`;
            return;
        }

        // List of semantic mapping items to mock concept highlights (e.g. synonyms)
        const semanticConcepts = [
            { words: ["pto", "days", "vacation", "off", "leave"], term: "vacation and pto policy" },
            { words: ["expense", "reimbursement", "meal", "travel", "bill", "concur"], term: "expense reimbursement guidelines" },
            { words: ["outage", "p0", "pagerduty", "page", "rotation", "incident", "call"], term: "engineering on-call rotation" },
            { words: ["mfa", "2fa", "password", "security", "credentials"], term: "security and password policy" },
            { words: ["code", "pr", "approvals", "merge", "pull"], term: "code review standards" },
            { words: ["remote", "hybrid", "telecommute", "online", "hours"], term: "hybrid work and attendance guidelines" },
            { words: ["speeds", "latency", "milestones", "goals", "indexing", "transformer"], term: "q3 search infrastructure goals" },
            { words: ["rag", "vector", "databases", "pyramid", "kadam"], term: "senior ml engineer" }
        ];

        elements.vectorResultsList.innerHTML = results.map((doc, idx) => {
            // Apply semantic concept highlight based on intent overlap
            let highlightedContent = doc.content;
            
            semanticConcepts.forEach(concept => {
                // If this is the relevant target document
                if (doc.title.toLowerCase().includes(concept.term) || doc.content.toLowerCase().includes(concept.term)) {
                    concept.words.forEach(word => {
                        const regex = new RegExp(`\\b(${word})\\b`, "gi");
                        highlightedContent = highlightedContent.replace(regex, `<span class="match-concept-hl">$1</span>`);
                    });
                }
            });

            const scorePercent = Math.round(doc.score * 100);

            return `
                <div class="result-card rank-${idx}">
                    <div class="card-top">
                        <span class="card-source ${doc.source}">✦ ${doc.source}</span>
                        <span class="card-score">Similarity: ${doc.score.toFixed(3)}</span>
                    </div>
                    <h4 class="card-title">${doc.title}</h4>
                    <p class="card-content">${highlightedContent}</p>
                    <div class="score-visual-container">
                        <div class="score-visual-bar" style="width: ${scorePercent}%"></div>
                    </div>
                </div>
            `;
        }).join("");
    }


    /* ==========================================================================
       TAB 2: RAG PIPELINE
       ========================================================================== */
    elements.ragSubmitBtn.addEventListener("click", runRAGPipeline);
    elements.ragQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") runRAGPipeline();
    });

    elements.ragStepTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            elements.ragStepTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const slideId = tab.getAttribute("data-step");
            elements.ragStepSlides.forEach(s => s.classList.remove("active"));
            document.getElementById(slideId).classList.add("active");
        });
    });

    async function runRAGPipeline() {
        const query = elements.ragQueryInput.value.trim();
        if (!query) return;

        // Clear previous state and show spinner/typing indicators
        elements.ragRetrievedChunks.innerHTML = `<div class="empty-state">Retrieving matching embeddings...</div>`;
        elements.ragPromptCode.textContent = `Formatting system instruction and prompt context...`;
        elements.ragResponseText.textContent = "";
        elements.ragResponseText.classList.add("empty");
        elements.ragTyping.style.display = "flex";

        // Auto move to step 1 Tab while executing
        triggerStepTab("output-step-retrieved");

        const data = await apiFetch(`/api/rag?q=${encodeURIComponent(query)}`);
        if (!data) return;

        // Render Step 1: Retrieved Chunks
        renderRAGChunks(data.retrieved_results);

        // Render Step 2: Prompt Code
        elements.ragPromptCode.textContent = data.prompt_template;

        // Render Step 3: Typewriter Synthesis
        elements.ragTyping.style.display = "none";
        elements.ragResponseText.classList.remove("empty");
        
        // Auto transition tabs in sequences
        setTimeout(() => {
            triggerStepTab("output-step-prompt");
            
            setTimeout(() => {
                triggerStepTab("output-step-response");
                typewriterEffect(elements.ragResponseText, data.generated_answer);
            }, 2500);
            
        }, 2000);
    }

    function triggerStepTab(slideId) {
        elements.ragStepTabs.forEach(t => {
            if (t.getAttribute("data-step") === slideId) {
                t.classList.add("active");
            } else {
                t.classList.remove("active");
            }
        });
        elements.ragStepSlides.forEach(s => {
            if (s.id === slideId) {
                s.classList.add("active");
            } else {
                s.classList.remove("active");
            }
        });
    }

    function renderRAGChunks(results) {
        if (!results || results.length === 0) {
            elements.ragRetrievedChunks.innerHTML = `<div class="empty-state">No context documents could be retrieved.</div>`;
            return;
        }

        elements.ragRetrievedChunks.innerHTML = results.slice(0, 3).map((doc, idx) => `
            <div class="retrieved-chunk-item">
                <div class="retrieved-chunk-header">
                    <span class="card-source ${doc.source}">Chunk #${idx + 1} | Source: ${doc.source}</span>
                    <span class="card-score">Similarity: ${doc.score.toFixed(3)}</span>
                </div>
                <h5 style="margin-bottom: 6px; font-weight:600;">${doc.title}</h5>
                <p style="font-size: 0.85rem; color:var(--text-secondary); line-height: 1.4;">${doc.content}</p>
            </div>
        `).join("");
    }

    function typewriterEffect(element, text) {
        if (state.ragTypingTimer) clearInterval(state.ragTypingTimer);
        
        element.innerHTML = "";
        const words = text.split(" ");
        let wordIndex = 0;

        state.ragTypingTimer = setInterval(() => {
            if (wordIndex < words.length) {
                element.innerHTML += words[wordIndex] + " ";
                wordIndex++;
            } else {
                clearInterval(state.ragTypingTimer);
            }
        }, 50); // Write one word every 50ms (extremely organic!)
    }


    /* ==========================================================================
       TAB 3: VECTOR VISUALIZER
       ========================================================================== */
    const canvas = elements.vectorCanvas;
    let ctx = null;
    let scaleFactor = 1;
    let offsetX = 0;
    let offsetY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    function initCanvas() {
        if (!ctx) {
            ctx = canvas.getContext("2d");
            
            // Set up high DPI canvas handling
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            // Attach mouse events
            canvas.addEventListener("mousemove", handleCanvasMouseMove);
            canvas.addEventListener("mousedown", handleCanvasMouseDown);
            canvas.addEventListener("mouseup", handleCanvasMouseUp);
            canvas.addEventListener("mouseleave", handleCanvasMouseLeave);
        }
    }

    elements.visQueryBtn.addEventListener("click", () => {
        state.visualizerQuery = elements.visQueryInput.value.trim();
        loadVisualizerData(false);
    });

    elements.visQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            state.visualizerQuery = elements.visQueryInput.value.trim();
            loadVisualizerData(false);
        }
    });

    elements.resetCanvasBtn.addEventListener("click", () => {
        scaleFactor = 1;
        offsetX = 0;
        offsetY = 0;
        drawVectorSpace();
    });

    async function loadVisualizerData(firstLoad = false) {
        let url = "/api/visualize";
        if (state.visualizerQuery) {
            url += `?q=${encodeURIComponent(state.visualizerQuery)}`;
        }

        const data = await apiFetch(url);
        if (!data) return;

        state.visualizerNodes = data.nodes;
        state.visualizerQueryCoord = data.query_coord;
        state.visualizerNearestNeighbors = data.nearest_neighbors;

        // Sync visualizer title mode
        if (data.engine_mode) {
            elements.engineModeText.textContent = data.engine_mode;
        }

        // Draw Canvas
        drawVectorSpace();

        // If not first load, select the query node in the UI details panel
        if (state.visualizerQueryCoord) {
            renderQueryDetailsPanel();
        }
    }

    // Mapping Cartesian [-80, 80] backend space to Canvas pixel layout
    function toCanvasCoords(x, y) {
        const dprRectWidth = canvas.width / (window.devicePixelRatio || 1);
        const dprRectHeight = canvas.height / (window.devicePixelRatio || 1);
        
        const cx = dprRectWidth / 2 + offsetX;
        const cy = dprRectHeight / 2 + offsetY;
        
        // Stretch multiplier to distribute dots nicely across 80% of canvas area
        const scale = (dprRectWidth / 2) * 0.75 * scaleFactor / 75;
        
        return {
            x: cx + x * scale,
            y: cy - y * scale // invert Y axis for traditional cartesian visualization
        };
    }

    // Map screen pixel location back to Cartwright space [-80, 80]
    function fromCanvasCoords(mx, my) {
        const dprRectWidth = canvas.width / (window.devicePixelRatio || 1);
        const dprRectHeight = canvas.height / (window.devicePixelRatio || 1);
        
        const cx = dprRectWidth / 2 + offsetX;
        const cy = dprRectHeight / 2 + offsetY;
        
        const scale = (dprRectWidth / 2) * 0.75 * scaleFactor / 75;
        
        return {
            x: (mx - cx) / scale,
            y: -(my - cy) / scale
        };
    }

    function drawVectorSpace() {
        const rectWidth = canvas.width / (window.devicePixelRatio || 1);
        const rectHeight = canvas.height / (window.devicePixelRatio || 1);
        
        // Clear canvas
        ctx.clearRect(0, 0, rectWidth, rectHeight);
        
        // 1. Draw Grid Lines (Cartesian Space background)
        drawGrid(rectWidth, rectHeight);

        // 2. Draw nearest neighbor linking beams (glow lines)
        if (state.visualizerQueryCoord && state.visualizerNearestNeighbors.length > 0) {
            const qPixel = toCanvasCoords(state.visualizerQueryCoord[0], state.visualizerQueryCoord[1]);
            
            state.visualizerNearestNeighbors.forEach(nodeId => {
                const node = state.visualizerNodes.find(n => n.id === nodeId);
                if (node) {
                    const nPixel = toCanvasCoords(node.x, node.y);
                    
                    // Draw neon link line
                    ctx.beginPath();
                    ctx.moveTo(qPixel.x, qPixel.y);
                    ctx.lineTo(nPixel.x, nPixel.y);
                    
                    // Style
                    const gradient = ctx.createLinearGradient(qPixel.x, qPixel.y, nPixel.x, nPixel.y);
                    gradient.addColorStop(0, "rgba(234, 179, 8, 0.4)");
                    gradient.addColorStop(1, "rgba(139, 92, 246, 0.1)");
                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([4, 4]); // Dash effect
                    ctx.stroke();
                    ctx.setLineDash([]); // Reset
                }
            });
        }

        // 3. Draw Document Nodes
        state.visualizerNodes.forEach(node => {
            const pixel = toCanvasCoords(node.x, node.y);
            let color = "#ff6e4a"; // Notion
            let shadowColor = "rgba(255, 110, 74, 0.5)";
            
            if (node.source === "Slack") {
                color = "#36c5f0";
                shadowColor = "rgba(54, 197, 240, 0.5)";
            } else if (node.source === "LinkedIn") {
                color = "#0077b5";
                shadowColor = "rgba(0, 119, 181, 0.5)";
            }

            const isHighlighted = state.hoveredNode && state.hoveredNode.id === node.id;
            const isNeighbor = state.visualizerNearestNeighbors.includes(node.id);

            ctx.beginPath();
            ctx.arc(pixel.x, pixel.y, isHighlighted ? 8 : (isNeighbor ? 6.5 : 5), 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.shadowColor = shadowColor;
            ctx.shadowBlur = isHighlighted ? 15 : (isNeighbor ? 10 : 4);
            ctx.fill();
            
            // Draw secondary outer ring for hover
            if (isHighlighted || isNeighbor) {
                ctx.beginPath();
                ctx.arc(pixel.x, pixel.y, isHighlighted ? 12 : 9, 0, 2 * Math.PI);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.shadowBlur = 0;
                ctx.stroke();
            }
        });

        // 4. Draw Query Vector node
        if (state.visualizerQueryCoord) {
            const qPixel = toCanvasCoords(state.visualizerQueryCoord[0], state.visualizerQueryCoord[1]);
            
            // Draw rotating star / diamond
            ctx.save();
            ctx.translate(qPixel.x, qPixel.y);
            ctx.rotate(Math.PI / 4); // 45 deg tilt for diamond
            
            ctx.beginPath();
            const size = 6;
            ctx.rect(-size, -size, size * 2, size * 2);
            ctx.fillStyle = "#eab308"; // warning gold
            ctx.shadowColor = "rgba(234, 179, 8, 0.8)";
            ctx.shadowBlur = 12;
            ctx.fill();
            ctx.restore();

            // Draw pulsing outer rings
            const pulse = (Date.now() % 2000) / 2000;
            ctx.beginPath();
            ctx.arc(qPixel.x, qPixel.y, 8 + pulse * 14, 0, 2 * Math.PI);
            ctx.strokeStyle = `rgba(234, 179, 8, ${1 - pulse})`;
            ctx.lineWidth = 1;
            ctx.shadowBlur = 0;
            ctx.stroke();
        }

        ctx.shadowBlur = 0; // reset shadow for future iterations
    }

    function drawGrid(width, height) {
        const cx = width / 2 + offsetX;
        const cy = height / 2 + offsetY;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
        ctx.lineWidth = 0.5;

        // Draw grid squares mapped to cartesian scale
        const spacing = 15 * scaleFactor;
        
        // Vertical lines
        for (let x = cx % spacing; x < width; x += spacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = cy % spacing; y < height; y += spacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw Central Axes
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 1;

        // Y Axis
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, height);
        ctx.stroke();

        // X Axis
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(width, cy);
        ctx.stroke();
    }

    // Handle mouse movement for node hover checks
    function handleCanvasMouseMove(e) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Find nearest document node within hover radius
        let foundNode = null;
        let minDistance = 12; // hover threshold in pixels

        state.visualizerNodes.forEach(node => {
            const pixel = toCanvasCoords(node.x, node.y);
            const dist = Math.hypot(pixel.x - mx, pixel.y - my);
            if (dist < minDistance) {
                minDistance = dist;
                foundNode = node;
            }
        });

        if (foundNode !== state.hoveredNode) {
            state.hoveredNode = foundNode;
            drawVectorSpace();
            
            if (state.hoveredNode) {
                renderNodeDetailsPanel(state.hoveredNode);
            } else if (state.visualizerQueryCoord) {
                renderQueryDetailsPanel();
            }
        }
    }

    // Drag to pan canvas functions
    function handleCanvasMouseDown(e) {
        isDragging = true;
        dragStartX = e.clientX - offsetX;
        dragStartY = e.clientY - offsetY;
    }

    function handleCanvasMouseUp() {
        isDragging = false;
    }

    function handleCanvasMouseLeave() {
        isDragging = false;
        state.hoveredNode = null;
        drawVectorSpace();
    }

    // Trigger pan logic on dragging
    canvas.addEventListener("mousemove", (e) => {
        if (isDragging) {
            offsetX = e.clientX - dragStartX;
            offsetY = e.clientY - dragStartY;
            drawVectorSpace();
        }
    });

    // Handle scroll wheel zoom
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const zoomIntensity = 0.05;
        const delta = e.deltaY < 0 ? 1 : -1;
        
        // Restrict scale scope
        scaleFactor = Math.max(0.5, Math.min(3, scaleFactor + delta * zoomIntensity));
        drawVectorSpace();
    });

    function renderNodeDetailsPanel(node) {
        const fullDoc = DOCUMENTS_LOCAL.find(d => d.id === node.id);
        
        elements.visNodeDetails.innerHTML = `
            <div class="node-details-header">
                <span class="card-source ${node.source}">✦ DOCUMENT NODE #${node.id}</span>
                <span class="card-score" style="background: rgba(255,255,255,0.05);">${node.source}</span>
            </div>
            <h4 class="node-details-title">${node.title}</h4>
            <p class="node-details-content">${fullDoc ? fullDoc.content : 'No content'}</p>
            <div style="font-size: 0.7rem; color:var(--text-muted); margin-top:12px; font-family:monospace">
                Coordinates: [x: ${node.x.toFixed(2)}, y: ${node.y.toFixed(2)}]
            </div>
        `;
    }

    function renderQueryDetailsPanel() {
        elements.visNodeDetails.innerHTML = `
            <div class="node-details-header">
                <span class="card-score" style="background: rgba(234, 179, 8, 0.15); color: #eab308">✦ QUERY VECTOR NODE</span>
                <span class="card-score">Active Query</span>
            </div>
            <h4 class="node-details-title" style="color:var(--warning)">"${state.visualizerQuery}"</h4>
            <p class="node-details-content">This query vector was projected into the 2D document feature space. Dotted neon lines link this query directly to its top-3 nearest neighbor documents based on cosine vector similarity.</p>
            <div style="font-size: 0.7rem; color:var(--text-muted); margin-top:12px; font-family:monospace">
                Coordinates: [x: ${state.visualizerQueryCoord[0].toFixed(2)}, y: ${state.visualizerQueryCoord[1].toFixed(2)}]
            </div>
        `;
    }

    // Keep active anim loop to pulse query dot nicely
    setInterval(() => {
        if (state.activeTab === "vector-visualizer" && state.visualizerQueryCoord) {
            drawVectorSpace();
        }
    }, 100);


    /* ==========================================================================
       TAB 4: RELEVANCE SIMULATOR
       ========================================================================== */
    elements.runBenchmarkBtn.addEventListener("click", triggerBenchmarkRun);
    elements.runBenchmarkBtnSub.addEventListener("click", triggerBenchmarkRun);

    async function triggerBenchmarkRun() {
        // Toggle empty state / details elements
        elements.benchmarkPreRun.style.display = "none";
        
        // Show simulated loading spinner or wait
        elements.benchmarkMetricsGrid.style.display = "grid";
        elements.benchmarkResultsContainer.style.display = "grid";

        elements.gainVal.textContent = "Calculating...";
        elements.vectorMrrVal.textContent = "---";
        elements.tfidfMrrVal.textContent = "---";
        elements.scenariosContainer.innerHTML = `<div class="empty-state">Executing 8 query scenarios and generating rankings...</div>`;

        const data = await apiFetch("/api/benchmark");
        if (!data) return;

        state.benchmarkData = data;

        // Set metrics values
        elements.gainVal.textContent = `+${data.mrr_gain_percent}%`;
        elements.vectorMrrVal.textContent = data.metrics.vector.mrr.toFixed(3);
        elements.tfidfMrrVal.textContent = data.metrics.tfidf.mrr.toFixed(3);

        // Render Chart.js visualization
        renderBenchmarkChart(data.metrics);

        // Render Scenarios accordion list
        renderBenchmarkScenarios(data.query_details);
    }

    function renderBenchmarkChart(metrics) {
        if (state.chartInstance) {
            state.chartInstance.destroy();
        }

        const ctxChart = elements.metricsChart.getContext("2d");
        state.chartInstance = new Chart(ctxChart, {
            type: "bar",
            data: {
                labels: ["Mean Reciprocal Rank (MRR)", "Precision @ 1", "Precision @ 3"],
                datasets: [
                    {
                        label: "TF-IDF Keyword Search",
                        data: [
                            metrics.tfidf.mrr,
                            metrics.tfidf.precision_at_1,
                            metrics.tfidf.precision_at_3
                        ],
                        backgroundColor: "rgba(148, 163, 184, 0.4)",
                        borderColor: "rgba(148, 163, 184, 0.8)",
                        borderWidth: 1,
                        borderRadius: 4
                    },
                    {
                        label: "Embedding Vector Search",
                        data: [
                            metrics.vector.mrr,
                            metrics.vector.precision_at_1,
                            metrics.vector.precision_at_3
                        ],
                        backgroundColor: "rgba(139, 92, 246, 0.6)",
                        borderColor: "rgba(139, 92, 246, 0.9)",
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: "#94a3b8",
                            font: { family: "Outfit", size: 12 }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: "#94a3b8", font: { family: "Outfit" } },
                        grid: { color: "rgba(255, 255, 255, 0.03)" }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        ticks: { color: "#94a3b8", font: { family: "Outfit" } },
                        grid: { color: "rgba(255, 255, 255, 0.03)" }
                    }
                }
            }
        });
    }

    function renderBenchmarkScenarios(scenarios) {
        elements.scenariosContainer.innerHTML = scenarios.map((s, index) => {
            const vWinner = s.vector_rank <= s.tfidf_rank || s.tfidf_rank === "Not Found (>5)";
            const tfidfRankText = s.tfidf_rank === "Not Found (>5)" ? ">5" : `#${s.tfidf_rank}`;
            const vectorRankText = s.vector_rank === "Not Found (>5)" ? ">5" : `#${s.vector_rank}`;

            return `
                <div class="scenario-item" data-index="${index}">
                    <div class="scenario-summary">
                        <span class="scenario-query">"${s.query}"</span>
                        <div class="ranks-badge">
                            <span class="rank-pill tfidf">TF-IDF: ${tfidfRankText}</span>
                            <span class="rank-pill vector ${vWinner ? 'winner' : ''}">Vector: ${vectorRankText}</span>
                        </div>
                    </div>
                    <div class="scenario-details" id="scenario-details-${index}">
                        <p><strong>Scenario Target Intent:</strong> ${s.intent}</p>
                        <p><strong>Target Document:</strong> "${s.target_doc_title}"</p>
                        <p><strong>Lexical Search Result:</strong> Ranked at index <strong>${tfidfRankText}</strong> (Top score: ${s.tfidf_score.toFixed(3)})</p>
                        <p><strong>Vector Search Result:</strong> Ranked at index <strong>${vectorRankText}</strong> (Similarity score: ${s.vector_score.toFixed(3)})</p>
                        <p style="margin-top: 8px; color: var(--text-primary); font-style: italic;">
                            💡 <strong>Mechanism:</strong> ${s.description}
                        </p>
                    </div>
                </div>
            `;
        }).join("");

        // Attach accordion click events
        const items = elements.scenariosContainer.querySelectorAll(".scenario-item");
        items.forEach(item => {
            item.addEventListener("click", () => {
                const idx = item.getAttribute("data-index");
                const details = document.getElementById(`scenario-details-${idx}`);
                
                // Toggle active details
                const currentlyActive = details.classList.contains("active");
                
                // Hide all details
                elements.scenariosContainer.querySelectorAll(".scenario-details").forEach(d => d.classList.remove("active"));
                
                if (!currentlyActive) {
                    details.classList.add("active");
                }
            });
        });
    }

    // Local copy of document titles & contents for visualizer details panel (safeguard)
    const DOCUMENTS_LOCAL = [
        { id: 1, title: "Vacation and PTO Policy", content: "Full-time employees receive 25 days of paid time off (PTO) annually. Requests must be submitted in Workday at least two weeks in advance. Unused PTO does not roll over to the next calendar year and expires on December 31st." },
        { id: 2, title: "Expense Reimbursement Guidelines", content: "Travel and business expenses must be logged in Concur. Meal expenses are capped at $75 per day. Receipts are required for all transactions over $25. Approved claims are paid out in the next monthly payroll cycle." },
        { id: 3, title: "Engineering On-Call Rotation", content: "The on-call rotation runs weekly starting Tuesday at 10:00 AM. The primary engineer is responsible for triaging P0 and P1 system outages. If unresponsive for 10 minutes, the incident escalates to the secondary on-call engineer via PagerDuty." },
        { id: 4, title: "Security and Password Policy", content: "All corporate accounts must use multi-factor authentication (MFA). Passwords must be at least 16 characters long, contain numbers and special characters, and be changed every 90 days. Sharing credentials in Slack is strictly prohibited." },
        { id: 5, title: "Code Review Standards", content: "All pull requests require at least two approvals from engineering team members before merging to the main branch. Ensure all automated unit tests pass and code coverage remains above 80%." },
        { id: 6, title: "Hybrid Work and Attendance Guidelines", content: "We support a hybrid work model. Employees can work remotely up to 3 days per week with manager approval. Core team hours are 10 AM to 4 PM EST, during which all members should be online and reachable." },
        { id: 7, title: "Q3 Search Infrastructure Goals", content: "Our primary Q3 objective is to improve search relevance scores by 29% and decrease latency by 30%. Key results include migrating to FAISS vector index, optimizing chunking strategies, and deploying dense sentence-transformer embeddings." },
        { id: 8, title: "alice: Expense report help", content: "hey team, does anyone know where the expense report form is? i checked workday but couldn't find it anywhere." },
        { id: 9, title: "bob: Re: Expense report help", content: "@alice check concur! all expense reports go through concur now. here is the link: concur.corp.internal. you'll need to upload the receipt there." },
        { id: 10, title: "charlie: Staging outage alert", content: "heads up, the staging server is throwing 500 errors since the last deploy. looking into it now. might be a database lock issue." },
        { id: 11, title: "dave: Re: Staging outage alert", content: "@charlie page the on-call engineer if it doesn't resolve in 10 minutes. who is on rotation this week? bob or eve?" },
        { id: 12, title: "eve: Re: Staging outage alert", content: "i think bob is the primary on-call engineer this week, let me double check the schedule in pagerduty. @bob are you online?" },
        { id: 13, title: "frank: PR styling design", content: "just submitted a new PR for the search page. added some cool glassmorphic styling and transition animations! let me know what you guys think." },
        { id: 14, title: "grace: MFA deadline notice", content: "remember to enable MFA on your slack account guys! the IT team is enforcing it today. if you don't set it up, you'll be locked out tomorrow." },
        { id: 15, title: "alice: Re: Expense report help", content: "thanks bob, concur worked perfectly. logged my $50 team dinner receipt there and it got approved instantly!" },
        { id: 16, title: "Omkar Kadam - Senior ML Engineer", content: "Omkar Kadam - Senior Machine Learning Engineer specializing in building scalable Retrieval-Augmented Generation (RAG) pipelines. Expert in PyTorch, sentence-transformers, and FAISS. Improved search relevance scores by 29% over TF-IDF in user simulation." },
        { id: 17, title: "Notion - Staff ML Engineer (Search)", content: "Staff Machine Learning Engineer role at Notion. Seeking experts in vector database design, semantic search, and large language model (LLM) fine-tuning. Help us build the future of organizational knowledge discovery!" },
        { id: 18, title: "Slack - Search Infrastructure Engineer II", content: "Software Engineer II - Search Infrastructure at Slack. Responsible for indexing billions of chat messages and delivering low-latency keyword and semantic search. Expertise in Elasticsearch, Lucene, and vector similarity retrieval." },
        { id: 19, title: "LinkedIn - NLP Research Scientist", content: "Machine Learning Researcher - Natural Language Processing at LinkedIn. Developing advanced representation learning and retrieval models for professional networking queries and job match relevance optimization." },
        { id: 20, title: "John Doe - Frontend Developer Portfolio", content: "John Doe - Software Developer. 5 years of experience in Django and vanilla JavaScript. Passionate about building responsive user interfaces with CSS grids, flexbox, and beautiful interactive micro-animations." },
        { id: 21, title: "Jane Smith - Product Manager, Notion AI", content: "Jane Smith - Product Manager at Notion. Led the launch of Notion AI and semantic search features, improving search click-through rate by 15% and user satisfaction scores." }
    ];

    /* ==========================================================================
       INITIAL SETUP AND BOOTSTRAPPING
       ========================================================================== */
    // Ping backend to load system mode and triggers initial search comparison
    setTimeout(() => {
        updateSystemState();
        runSearchPlayground();
    }, 1000);
});
