"""
Streamlit Frontend for OCR-Aware Question Solving Agent.
Visualizes the step-by-step retry, option matching, and visual OCR refinement loop.
"""

import json
import logging
from pathlib import Path
import tempfile
from typing import Optional

import streamlit as st
from PIL import Image

from ocr_solver.agent.loop import OCRAwareSolverAgent
from ocr_solver.agent.options import clean_text, extract_options_from_text
from ocr_solver.config import settings
from ocr_solver.formatter import format_single_result
from ocr_solver.models import DetailedSolverResult, QuestionOutput, SolveAttempt
from ocr_solver.noise.persian_noise import PersianOCRNoiseInjector
from ocr_solver.ocr.datalab import DatalabOCRClient

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="OCR-Aware Solver Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-info {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_available_samples():
    """Retrieve existing sample images in the samples/ directory."""
    samples_dir = Path("samples")
    if samples_dir.exists():
        return sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    return []


def run_solver_with_live_timeline(
    image_path: Path,
    inject_noise: bool = False,
    noise_rate: float = 0.08,
    max_perturbations: int = 3,
    max_retries: int = 3,
) -> DetailedSolverResult:
    """
    Executes the OCR-aware solver pipeline step-by-step while rendering
    live progress and thought processes into st.status and expandable containers.
    """
    ocr_client = DatalabOCRClient()
    noise_injector = PersianOCRNoiseInjector()
    agent = OCRAwareSolverAgent(max_retries=max_retries)

    with st.status("🧠 Agent is analyzing and solving the question...", expanded=True) as status:
        # ---------------------------------------------------------
        # Step 1: Extract Initial OCR
        # ---------------------------------------------------------
        st.write("### 📄 Step 1: Extracting Initial OCR")
        ocr_res = ocr_client.extract_text(image_path)
        initial_text = ocr_res.text
        original_ocr = initial_text

        with st.expander("👁️ View Initial Extracted OCR Text", expanded=False):
            st.code(initial_text, language="markdown")
            st.caption(f"Provider: **{ocr_res.provider}** | Status: **{'Success' if ocr_res.success else 'Failed'}**")

        # Step 1b: Optional Noise Injection for Stress-Testing
        if inject_noise:
            st.write("### ⚡ Step 1b: Synthetic Noise Injection (Stress-Testing)")
            corrupted_text, perts = noise_injector.perturb_text(
                initial_text,
                char_rate=noise_rate,
                max_perturbations=max_perturbations,
            )
            with st.expander(f"⚠️ Perturbations Injected ({len(perts)} mutations)", expanded=True):
                for p in perts:
                    st.markdown(
                        f"- Mutated **`{p.original_char}`** $\\to$ **`{p.perturbed_char}`** "
                        f"*(Category: `{p.category}`)* at index {p.position}"
                    )
                st.markdown("**Corrupted OCR Text fed to Agent:**")
                st.code(corrupted_text, language="markdown")
            initial_text = corrupted_text

        # ---------------------------------------------------------
        # Step 2: Initial Solve Attempt
        # ---------------------------------------------------------
        st.write("### 📐 Step 2: Initial LLM Solve Attempt")
        current_text = initial_text
        attempts: list[SolveAttempt] = []
        is_changed = False
        options = extract_options_from_text(current_text)

        attempt_num = 1
        attempt_result = agent._execute_solve_attempt(
            image_path=image_path,
            question_text=current_text,
            attempt_number=attempt_num,
            is_correction=False,
        )
        attempts.append(attempt_result)

        with st.expander(f"📝 Attempt {attempt_num} Reasoning & Computation", expanded=True):
            st.markdown(f"**Parsed Options:** `{options}`")
            st.markdown(f"**Mathematical Reasoning:**\n\n{attempt_result.reasoning}")
            st.markdown(f"**Computed Value:** `{attempt_result.computed_value}`")

        # ---------------------------------------------------------
        # Step 3: Option Matching Verification
        # ---------------------------------------------------------
        st.write("### 🎯 Step 3: Option Matching Verification")
        if attempt_result.matched_option:
            st.success(
                f"✅ **Clean Match on Attempt 1!** Computed value matches **Option {attempt_result.matched_option}** "
                f"(Confidence: {attempt_result.match_confidence:.0%})"
            )
            status.update(
                label=f"✅ Solved successfully on Attempt 1 (Answer: Option {attempt_result.matched_option})",
                state="complete",
                expanded=False,
            )
            return DetailedSolverResult(
                output=QuestionOutput(
                    answer=attempt_result.matched_option,
                    question_text=current_text,
                    changed=False,
                    original_ocr_text=original_ocr,
                ),
                is_resolved=True,
                total_attempts=1,
                attempts=attempts,
                image_path=str(image_path),
            )

        st.warning(
            f"❌ **Option Mismatch Detected!** Derived value `{attempt_result.computed_value}` "
            f"did not match any candidate option in `{list(options.keys())}`. Transcription error suspected."
        )

        # ---------------------------------------------------------
        # Step 4: Visual OCR Refinement and Retry Loop
        # ---------------------------------------------------------
        while attempt_num <= agent.max_retries:
            st.write(f"### 🔄 Step 4: Visual OCR Refinement & Retry (Iteration {attempt_num})")
            
            # Step 4a: Visual Inspection
            correction_res = agent._refine_ocr_with_vision(
                image_path=image_path,
                current_text=current_text,
                previous_attempt=attempts[-1],
            )
            corrected_text = correction_res.get("corrected_question_text") or current_text
            notes = correction_res.get("correction_notes", "")
            errors = correction_res.get("identified_errors", [])

            with st.expander(f"🔍 Visual Inspection & Error Analysis (Retry {attempt_num})", expanded=True):
                st.markdown(f"**Identified Errors:** {errors if errors else 'Minor transcription discrepancy'}")
                st.markdown(f"**Correction Notes:** {notes}")
                
                # Check text diff
                if clean_text(corrected_text) != clean_text(current_text):
                    is_changed = True
                    current_text = corrected_text
                    options = extract_options_from_text(current_text)
                    st.info("💡 **Question text was corrected based on image re-examination:**")
                    st.code(current_text, language="markdown")
                else:
                    st.info("ℹ️ Question text preserved after visual re-examination.")

            attempt_num += 1

            # Step 4b: Re-solve with corrected text
            st.write(f"### 📐 Re-Solving with Corrected Input (Attempt {attempt_num})")
            attempt_result = agent._execute_solve_attempt(
                image_path=image_path,
                question_text=current_text,
                attempt_number=attempt_num,
                is_correction=True,
                correction_notes=notes,
            )
            attempts.append(attempt_result)

            with st.expander(f"📝 Attempt {attempt_num} Reasoning", expanded=True):
                st.markdown(f"**Reasoning:**\n\n{attempt_result.reasoning}")
                st.markdown(f"**Computed Value:** `{attempt_result.computed_value}`")

            # Check matching
            if attempt_result.matched_option:
                st.success(
                    f"🎉 **Refinement Successful on Attempt {attempt_num}!** "
                    f"Matched **Option {attempt_result.matched_option}** (Confidence: {attempt_result.match_confidence:.0%})"
                )
                status.update(
                    label=f"✅ Resolved on Attempt {attempt_num} after visual refinement (Answer: Option {attempt_result.matched_option})",
                    state="complete",
                    expanded=False,
                )
                return DetailedSolverResult(
                    output=QuestionOutput(
                        answer=attempt_result.matched_option,
                        question_text=current_text,
                        changed=is_changed,
                        original_ocr_text=original_ocr,
                    ),
                    is_resolved=True,
                    total_attempts=attempt_num,
                    attempts=attempts,
                    image_path=str(image_path),
                )

        # ---------------------------------------------------------
        # Fallback if retry cap reached
        # ---------------------------------------------------------
        st.write("### ⚖️ Step 5: Bounded Fallback Resolution")
        st.warning(f"Retry cap of {agent.max_retries} reached. Engaging best-guess heuristic fallback.")
        best_guess, fallback_reason = agent._resolve_fallback(
            image_path=image_path,
            question_text=current_text,
            options=options,
            attempts=attempts,
        )

        with st.expander("🛡️ Fallback Heuristic Justification", expanded=True):
            st.markdown(f"**Selected Best-Guess Option:** `{best_guess}`")
            st.markdown(f"**Justification:** {fallback_reason}")

        status.update(
            label=f"⚠️ Concluded with Best-Guess Option {best_guess} (Retry cap reached)",
            state="complete",
            expanded=False,
        )

        return DetailedSolverResult(
            output=QuestionOutput(
                answer=best_guess,
                question_text=current_text,
                changed=is_changed,
                original_ocr_text=original_ocr,
            ),
            is_resolved=False,
            total_attempts=len(attempts),
            attempts=attempts,
            image_path=str(image_path),
            unresolved_reason=fallback_reason,
        )


def main():
    # Header
    st.markdown('<div class="main-title">🔍 OCR-Aware Question Solving Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Visualizing the self-correcting OCR refinement and option matching workflow for Persian STEM exams.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration & Input")
        
        input_source = st.radio(
            "Select Question Image Source:",
            ["Sample Questions", "Upload Custom Image"],
            index=0,
        )

        image_path: Optional[Path] = None
        uploaded_file = None

        if input_source == "Sample Questions":
            sample_files = get_available_samples()
            if sample_files:
                sample_names = [f.name for f in sample_files]
                selected_sample = st.selectbox("Choose a sample question:", sample_names, index=0)
                image_path = Path("samples") / selected_sample
            else:
                st.warning("No files found in samples/ directory.")
        else:
            uploaded_file = st.file_uploader(
                "Upload question crop (PNG/JPG):",
                type=["png", "jpg", "jpeg"],
                help="Upload a cropped image of a single multiple-choice question.",
            )
            if uploaded_file is not None:
                # Save uploaded file to a temporary file
                suffix = Path(uploaded_file.name).suffix or ".png"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(uploaded_file.read())
                temp_file.flush()
                image_path = Path(temp_file.name)

        st.divider()
        st.subheader("🧪 Synthetic Noise Test (Phase 3)")
        inject_noise = st.checkbox(
            "Inject Synthetic Persian OCR Noise",
            value=False,
            help="Simulates realistic Persian character confusion and OCR degradation to stress-test error recovery.",
        )
        noise_rate = 0.08
        max_noise_perturbations = 3
        if inject_noise:
            noise_rate = st.slider("Character Corruption Rate", 0.02, 0.30, 0.08, 0.02)
            max_noise_perturbations = st.slider("Max Perturbations", 1, 6, 3, 1)

        st.divider()
        st.subheader("🤖 Agent Settings")
        max_retries = st.number_input("Max Refinement Retries", min_value=1, max_value=5, value=settings.MAX_RETRIES)
        st.caption(f"LLM Model: `{settings.llm_model}`")
        st.caption(f"Mock Fallback: `{settings.OCR_MOCK_FALLBACK}`")

    # Main Area Layout
    col_left, col_right = st.columns([1, 1.4], gap="medium")

    with col_left:
        st.subheader("🖼️ Question Image")
        if image_path and image_path.exists():
            try:
                img = Image.open(image_path)
                st.image(img, caption=f"Selected: {image_path.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.info("👈 Please select or upload a question image from the sidebar to begin.")

    with col_right:
        st.subheader("🚀 Execution & Workflow Log")

        if image_path and image_path.exists():
            run_button = st.button("▶️ Solve Question", type="primary", use_container_width=True)

            if run_button:
                # Run the step-by-step solver
                result = run_solver_with_live_timeline(
                    image_path=image_path,
                    inject_noise=inject_noise,
                    noise_rate=noise_rate,
                    max_perturbations=max_noise_perturbations,
                    max_retries=max_retries,
                )

                st.divider()
                st.subheader("📊 Final Output & Metrics")

                # Metrics Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="Selected Answer", value=f"Option {result.output.answer}")
                with m2:
                    st.metric(label="Total Attempts", value=result.total_attempts)
                with m3:
                    changed_badge = "Yes (Corrected)" if result.output.changed else "No (Original)"
                    st.metric(label="OCR Text Changed", value=changed_badge)

                # Strict Output Schema
                st.markdown("#### 📦 Final Parsed JSON Output (Phase 4 Schema)")
                final_json_dict = format_single_result(result)
                st.json(final_json_dict)

                # Download button
                json_str = json.dumps(final_json_dict, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 Download Result JSON",
                    data=json_str,
                    file_name=f"result_{image_path.stem}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.write("Awaiting image selection...")


if __name__ == "__main__":
    main()
