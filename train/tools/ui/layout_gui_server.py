from nicegui import ui, run
import os
import fitz
import base64
from io import BytesIO
import threading

import pymupdf.layout.DocumentLayoutAnalyzer as DocumentLayoutAnalyzer

# -----------------------------------------------------------
#  Model Singleton Class
# -----------------------------------------------------------
class ModelManager:
    """
    Implements the Singleton pattern to ensure models (da_cur, da_dev) are
    loaded only once and reused across different threads/calls.
    """
    _instance = None
    _lock = threading.Lock()
    _models = {}  # Cache for loaded models

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check inside the lock
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    # Initialize models upon first instance creation
                    cls._instance._load_initial_models()
        return cls._instance

    def _load_initial_models(self):
        """Initial model loading logic."""
        print("Loading models (Singleton initialization)...")

        # Load da_cur
        try:
            da_cur = DocumentLayoutAnalyzer.get_model()
            self._models['cur'] = da_cur
            print("Successfully loaded 'cur' model.")
        except Exception as e:
            self._models['cur'] = None
            print(f"Error loading 'cur' model: {e}")

    def get_model(self, da_type: str):
        """Returns the cached model instance for the given type."""
        model = self._models.get(da_type)
        if model is None:
            raise ValueError(f"Model '{da_type}' is not loaded or failed to load.")
        return model


# -----------------------------------------------------------

da_type = 'cur'
current_data_dict = None

# --- Configuration ---
UPLOAD_DIR = './temp/uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Shared State (State added and modified) ---
current_doc = None  # fitz.Document object (only used in the main thread for length check/quick access)
current_pdf_path = None  # Store file path for background processes (CPU-bound tasks)
current_page_index = 0
color_map = {}
is_rendering: bool = False  # Rendering progress status flag
rendered_html_content: str = '<p>No PDF loaded. Please upload a PDF file on the control panel.</p>'  # Variable to store the rendering result
layout_mode: str = 'Layout'  # 'Node' or 'Layout'. Default is 'Layout'.


def get_layout_result(doc, page_no, da_type, return_bbox=False):
    """
    Perform layout prediction and return normalized bounding boxes (Layout result)
    and the raw input data (Node result).

    da_type (str): 'dev' or 'cur' to select the model.

    Returns: (det_result: list, data_dict: dict)
    """
    # ?? Get model instance from Singleton Manager
    try:
        da = ModelManager().get_model(da_type)
    except ValueError as e:
        print(f"Model retrieval error: {e}")
        return [], {'bboxes': []}

    # Model-specific data extraction logic
    if da_type == 'dev':
        from train.infer.pymupdf_util import create_input_data_from_page
        data_dict = create_input_data_from_page(doc[page_no])
    elif da_type == 'cur':
        from pymupdf.layout.pymupdf_util import create_input_data_from_page
        data_dict = create_input_data_from_page(doc[page_no])
    else:
        # Should not happen if ModelManager is working correctly
        return [], {'bboxes': []}

    # data_dict['bboxes'] contains ungrouped node boxes [x1, y1, x2, y2].
    if len(data_dict['bboxes']) == 0 or return_bbox:
        # Return empty result and the data_dict for the node mode
        return [], data_dict
    else:
        # det_result contains grouped boxes [x1, y1, x2, y2, cls_name].
        det_result = da.predict(data_dict)
        return det_result, data_dict


def get_classes_for_color_mapping(pdf_path: str, da_type: str):
    """
    Scan the first few pages (up to 5) of the document to collect all unique layout classes.
    This function opens the document internally to avoid pickling the fitz.Document object.

    da_type (str): 'dev' or 'cur' to select the model for class extraction.
    """
    classes = set()
    try:
        doc = fitz.open(pdf_path)  # Open inside the separate process
    except Exception:
        return []

    for page_num in range(min(5, len(doc))):  # Scan only the first 5 pages
        # Call get_layout_result and unpack the result. We only need the det_result here.
        det_result, _ = get_layout_result(doc, page_num, da_type)
        for *_, cls in det_result:
            classes.add(cls)

    doc.close()  # Essential to close the document opened in this process
    return list(classes)


def _render_page_html_sync(page_idx: int, pdf_path: str, color_map: dict, layout_mode: str, da_type: str) -> str:
    """
    Perform layout prediction and render the result as HTML/SVG based on the selected mode.
    The surrounding div applies Mac-style shadow and rounded corners.
    """
    # ... (PDF loading, pixmap creation, HEAVY CALCULATION logic remains unchanged)

    if not pdf_path:
        return '<p>No PDF path available.</p>', []

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return '<p>Error reopening PDF document in background process.</p>', []

    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return '<p>Invalid page index.</p>', []

    zoom = 1.5
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_buffer = BytesIO(pix.tobytes("png"))
    encoded_image = f"data:image/png;base64,{base64.b64encode(img_buffer.read()).decode()}"

    # === HEAVY CALCULATION ===
    det_result, data_dict = get_layout_result(doc, page_idx, da_type, return_bbox=layout_mode == 'Node')

    # --- Result Selection Logic ---
    if layout_mode == 'Node':
        display_results = [(x1, y1, x2, y2, 'Node') for x1, y1, x2, y2 in data_dict['bboxes']]
        default_color = "#808080"
    else:  # 'Layout' mode (Default)
        display_results = det_result
        default_color = "#ff0000"

    width, height = pix.width, pix.height

    svg_elements = []
    for i, (x1, y1, x2, y2, cls) in enumerate(display_results):
        px1, py1 = int(x1 * zoom), int(y1 * zoom)
        px2, py2 = int(x2 * zoom), int(y2 * zoom)

        if layout_mode == 'Node':
            color = default_color
            label = 'Node'
        else:
            color = color_map.get(cls, default_color)
            label = cls

        # --- Modified Rendering Logic ---
        if layout_mode == 'Node':
            # Node Mode: Clickable, No Text Label
            rect_attribs = (
                f'onclick="this.dispatchEvent(new CustomEvent(\'rect_click\', {{ bubbles: true, detail: {{ index: {i} }} }}))" '
                f'style="fill:none;stroke:{color};stroke-width:1;filter:drop-shadow(0 0 1px {color}); cursor: pointer; pointer-events: all;"'
            )
            text_element = ""
        else:
            # Layout Mode: Not Clickable, Show Text Label
            rect_attribs = (
                f'style="fill:none;stroke:{color};stroke-width:2;filter:drop-shadow(0 0 1px {color}); pointer-events: none;"'
            )

            label_y = max(py1 - 8, 20)
            text_element = f"""
                <text x="{px1 + 5}" y="{label_y}" fill="{color}" font-weight="bold" font-size="14"
                    style="text-shadow: 1px 1px 1px #000000; pointer-events: none;">{label}</text>
            """

        svg_elements.append(f"""
            <rect x="{px1}" y="{py1}" width="{px2 - px1}" height="{py2 - py1}" {rect_attribs} />
            {text_element}
        """)

    doc.close()

    # Apply Mac-style shadow (softer, wider) and rounded corners to the image container
    html_content = f"""
        <div style="position:relative;width:100%;max-width:{width}px;margin:auto;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1); /* Softer, Mac-like shadow */
                    border-radius: 12px; /* Smooth rounded corners */
                    overflow: hidden;"> 
            <img src="{encoded_image}" style="width:100%;display:block;border-radius: 12px;" />
            <svg width="100%" height="100%" viewBox="0 0 {width} {height}"
                    style="position:absolute;top:0;left:0;">
                {''.join(svg_elements)}
            </svg>
        </div>
    """
    return html_content, data_dict


# -----------------------------------------------------------
# Start asynchronous rendering task (manage UI updates)
# -----------------------------------------------------------
async def trigger_render():
    """Starts the CPU-bound layout analysis task in a background thread."""
    global current_doc, current_page_index, is_rendering, rendered_html_content, current_pdf_path, color_map, layout_mode, da_type, current_data_dict

    if current_doc is None:
        return

    # Prevent duplicate execution if already rendering
    if is_rendering:
        ui.notify('Render is already in progress.', color='warning')
        return

    page_idx = current_page_index
    current_mode = layout_mode  # Capture current mode for the rendering task
    current_da_type = da_type  # Capture current model type
    is_rendering = True

    # Immediately update with loading status message upon starting render
    loading_message = f'<div class="flex flex-col items-center mt-20"><div class="q-spinner-dots q-spinner" style="font-size: 50px; color: #1976d2;"></div><p class="mt-4 text-lg text-gray-600">Analyzing page {page_idx + 1} ({current_mode} Mode / Model: {current_da_type})... Please wait.</p></div>'
    rendered_html_content = loading_message

    try:
        # Execute the heavy synchronous function in a separate thread using run.cpu_bound
        # Pass current_da_type to the sync function
        new_content, data_dict = await run.cpu_bound(_render_page_html_sync, page_idx, current_pdf_path, color_map, current_mode, current_da_type)

        # Check if the user changed the page or model type while the task was completing
        if current_page_index == page_idx and layout_mode == current_mode and da_type == current_da_type:
            rendered_html_content = new_content
            current_data_dict = data_dict
        else:
            # Notify but don't overwrite if state changed (page, mode, or model type)
            ui.notify(
                f'Rendered page {page_idx + 1} ({current_mode} Mode, {current_da_type} Model) finished, but state changed.',
                color='info')


    except Exception as e:
        rendered_html_content = f'<p class="text-red-500 font-bold p-10 bg-red-100 rounded-lg shadow-md">Error rendering page {page_idx + 1}: {e}</p>'
        print(f"Rendering error: {e}")
        current_data_dict = {}
    finally:
        is_rendering = False


# -----------------------------------------------------------
# Update page navigation function
# -----------------------------------------------------------
async def change_page(direction: int):
    """Change the currently displayed page by the given direction (¡¾1) and trigger render."""
    global current_doc, current_page_index
    if not current_doc:
        ui.notify('No PDF loaded', color='negative')
        return

    new_idx = current_page_index + direction
    if 0 <= new_idx < len(current_doc):
        current_page_index = new_idx
        await trigger_render()
    else:
        ui.notify('Reached the beginning or end of the document.', color='warning')


# --- PAGE: Viewer ---
@ui.page('/viewer')
def viewer_page():
    """
    Viewer page: displays the currently selected PDF page with layout overlays.
    Implements independent scrolling for the PDF Viewer (left) and the Node Properties Panel (right).
    """
    global rendered_html_content, is_rendering, current_data_dict

    # w-full h-full no-wrap: Main row takes full height and prevents wrapping
    with ui.row().classes('w-full h-full no-wrap'):

        # --- Left Column: PDF Viewer (66%) ---
        # The main PDF viewer area, which is itself scrollable if content overflows
        with ui.column().classes('w-[66%] h-full p-4 overflow-hidden'):
            viewer_container = ui.column().classes('w-full h-full overflow-y-auto')
            with viewer_container:
                viewer_html = ui.html(rendered_html_content, sanitize=False).classes('w-full')

        # --- Right Column: Property Panel (34%) ---
        # Parent container for the panel content. Must be h-full to anchor the scrollable area.
        with ui.column().classes('w-[34%] h-full border-l border-gray-200 bg-gray-50 p-4 shadow-inner'):

            # Static Header: Will stay fixed at the top
            ui.label('Node Properties').classes('text-xl font-bold mb-4 text-gray-800 border-b pb-2 w-full')

            # Scrollable Wrapper: Takes the remaining vertical space (h-full) and manages scroll (overflow-y-auto).
            # This ensures the scroll is isolated to this section.
            with ui.column().classes('w-full h-full overflow-y-auto'):
                # Container for dynamic property rows (Content will push this to scroll)
                properties_container = ui.column().classes('w-full gap-2')

                # Initial placeholder text
                with properties_container:
                    ui.label('Select a node in "Node Mode" to view details.').classes('text-gray-500 italic')

        # --- Event Handler ---
        def handle_rect_click(e):
            """
            Handles click events from SVG rectangles.
            Populates the property panel with data in a clean, dense table format.
            """
            try:
                # Clear previous properties
                properties_container.clear()
                idx = e.args['detail']['index']

                # Validation checks (unchanged)
                if not current_data_dict or 'custom_features' not in current_data_dict or \
                        idx >= len(current_data_dict['custom_features']):
                    with properties_container:
                        ui.label('No feature data or invalid index.').classes('text-red-500')
                    return

                # Retrieve features
                custom_features = current_data_dict['custom_features'][idx]
                text_val = current_data_dict['text'][idx]

                # Populate Property Panel
                with properties_container:
                    ui.label(f'[{idx}]  {text_val}').classes('font-bold text-blue-600 mb-2')

                    # Define table columns (unchanged)
                    columns = [
                        {'name': 'key', 'label': 'Key', 'field': 'key', 'required': True, 'align': 'left',
                         'sortable': True},
                        {'name': 'value', 'label': 'Value', 'field': 'value', 'required': True, 'align': 'left',
                         'sortable': True},
                    ]

                    # Format features into table rows
                    rows = [{'key': key, 'value': str(value)} for key, value in custom_features.items()]

                    # Display table: flat, bordered, and dense for a compact look.
                    ui.table(
                        columns=columns,
                        rows=rows,
                        row_key='key'
                    ).props('flat bordered dense wrap-cells').classes('w-full')

            except Exception as err:
                print(f"Click handling error: {err}")
                with properties_container:
                    ui.label(f"Error: {err}").classes('text-red-500')

        # Bind the click event
        viewer_html.on('rect_click', handle_rect_click)

        # Timer to refresh viewer content (unchanged)
        def update_viewer_content():
            if viewer_html.content != rendered_html_content:
                viewer_html.content = rendered_html_content

        ui.timer(0.2, update_viewer_content)


@ui.page('/control')
def control_page():
    """Control page: handles file upload, navigation, and page input with Mac-style UI."""
    global current_page_index, current_doc, color_map, layout_mode, da_type

    # Use clean white background and generous padding.
    with ui.column().classes('w-full h-full items-center justify-start p-6 bg-white'):

        # ... (handle_upload function remains unchanged)

        async def handle_upload(e):
            """Handle PDF file upload or re-analysis trigger."""
            # ... (Implementation remains unchanged)
            global current_doc, color_map, current_page_index, current_pdf_path, da_type

            if hasattr(e, 'file') and e.file.name:
                pdf_path = os.path.join(UPLOAD_DIR, e.file.name)
                content = await e.file.read()
                with open(pdf_path, 'wb') as f:
                    f.write(content)

                current_pdf_path = pdf_path
                try:
                    current_doc = fitz.open(pdf_path)
                except Exception as load_error:
                    ui.notify(f"Error loading PDF: {load_error}", color='negative')
                    current_doc = None
                    current_pdf_path = None
                    return

                ui.notify(f"Loaded {len(current_doc)} pages. Analyzing layouts...", color='positive')
                current_page_index = 0

            elif current_pdf_path and current_doc:
                pdf_path = current_pdf_path
                ui.notify(f"Re-analyzing page {current_page_index + 1} with current model ({da_type.upper()})...",
                          color='info')
            else:
                return

            # Color mapping logic remains unchanged
            COLORS = [
                "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
                "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
                "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
            ]
            PREDEFINED_CLASSES = {
                'text': COLORS[0], 'title': COLORS[1], 'picture': COLORS[2], 'table': COLORS[3], 'list-item': COLORS[4],
                'page-header': COLORS[5], 'page-footer': COLORS[6], 'section-header': COLORS[7], 'footnote': COLORS[8],
                'caption': COLORS[9], 'formula': COLORS[10], 'unlabelled': COLORS[11],
            }
            color_map.clear()
            color_map.update(PREDEFINED_CLASSES)
            color_index = len(PREDEFINED_CLASSES)

            if len(current_doc) > 0:
                try:
                    all_classes = await run.cpu_bound(get_classes_for_color_mapping, current_pdf_path, da_type)
                    for cls in all_classes:
                        if cls not in color_map:
                            if color_index < len(COLORS):
                                color_map[cls] = COLORS[color_index]
                                color_index += 1
                            else:
                                color_map[cls] = "#000000"
                except Exception as color_map_error:
                    print(f"Error during color mapping analysis: {color_map_error}")
                    ui.notify("Error analyzing initial pages for colors.", color='negative')

            await trigger_render()

        # File Upload: Use flat design and rounded corners for a cleaner look.
        ui.upload(on_upload=handle_upload, label='Upload PDF') \
            .props('accept=".pdf" flat color="primary"').classes('w-full my-4 rounded-xl')

        # Separator: Thinner and lighter.
        ui.separator().classes('w-full opacity-50')

        # Layout Mode Selection: Minimal styling.
        async def update_layout_mode(e):
            """Update global layout_mode and trigger a re-render."""
            global layout_mode
            layout_mode = e.value
            ui.notify(f"View Mode set to **{layout_mode}**.", color='info')
            if current_doc:
                await trigger_render()

        ui.label('Layout View Mode:').classes('text-sm font-semibold mt-2 text-gray-700')
        # Use primary color for radio buttons
        ui.radio(
            ['Layout', 'Node'],
            value=layout_mode,
            on_change=update_layout_mode,
        ).props('inline color="blue"').classes('mt-1 mb-4')

        ui.separator().classes('w-full opacity-50')

        # Page navigation buttons: Use 'flat' and 'rounded' for circular, minimal buttons.
        with ui.row().classes('gap-4 mt-6'):
            ui.button('Prev', on_click=lambda: change_page(-1)).props('icon="arrow_back_ios" flat rounded').classes(
                'w-20')
            ui.label().classes('w-4')
            ui.button('Next', on_click=lambda: change_page(1)).props('icon="arrow_forward_ios" flat rounded').classes(
                'w-20')

        # Direct page input: Use 'outlined' style and rounded corners.
        with ui.row().classes('items-center mt-4 gap-2'):
            # Apply 'outlined' style and 'rounded-lg' class
            page_input = ui.number(label='Go to Page', min=1, value=1).props('outlined rounded-lg').classes('w-32')

            # ... (user_editing logic remains unchanged)
            user_editing = False

            def on_input_change(e):
                """Mark that user is editing the input."""
                nonlocal user_editing
                user_editing = True

            page_input.on('keydown', on_input_change)

            async def go_to_page():
                """Go to the specific page number entered by the user and trigger render."""
                global current_doc, current_page_index
                nonlocal user_editing
                user_editing = False

                if not current_doc:
                    ui.notify('No PDF loaded', color='negative')
                    return

                page_num = int(page_input.value)
                if 1 <= page_num <= len(current_doc):
                    current_page_index = page_num - 1
                    await trigger_render()
                else:
                    ui.notify(f'Invalid page number (1 - {len(current_doc)})', color='negative')

            # 'Go' button: flat and rounded to match navigation buttons
            ui.button('Go', on_click=go_to_page).props('icon="send" flat rounded')

        # Current page status display
        page_label = ui.label('No PDF').classes('mt-4 text-xl font-bold text-blue-800')

        # ... (update_page_label function remains unchanged)
        def update_page_label():
            """Keep page number display and input synchronized with the viewer."""
            global current_doc, current_page_index, is_rendering, layout_mode, da_type
            if current_doc:
                status = f" ({layout_mode} Mode, {da_type.upper()} Model)"
                if is_rendering:
                    status += " - Analyzing..."

                page_label.text = f'Page {current_page_index + 1} / {len(current_doc)}{status}'

                if not user_editing:
                    page_input.value = current_page_index + 1
            else:
                page_label.text = 'No PDF loaded'

        ui.timer(0.5, update_page_label)


# --- PAGE: Main Split Layout ---
@ui.page('/')
def main_page():
    """Main layout: displays control panel and viewer in a split-screen with Mac-style aesthetics."""
    ui.add_head_html('<title>PDF Visualizer</title>')

    # Split layout (25% / 75%) with clean, bright background and subtle divider shadow.
    ui.html("""
    <div style="display:flex;width:100vw;height:100vh;overflow:hidden;background-color:#ffffff;">
        <iframe src="/control" style="width:25%;height:100%;border:none;background-color:white;
                                     box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);"></iframe>
        <iframe src="/viewer" style="width:75%;height:100%;border:none;"></iframe>
    </div>
    """, sanitize=False)


# --- RUN SERVER ---
# Ensure ModelManager is initialized before the server starts to load models on the main thread
ModelManager()
ui.run(title='PDF Visualizer', port=11434, host='0.0.0.0', reload=True)