"""
Streamlit Frontend for OCR-Aware Question Solving Agent.
Visualizes the step-by-step retry, option matching, and visual OCR refinement loop with high UX transparency.
"""

import json
import logging
from pathlib import Path
import tempfile
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from ocr_solver.agent.loop import OCRAwareSolverAgent
from ocr_solver.agent.options import clean_text, extract_options_from_text
from ocr_solver.config import settings
from ocr_solver.formatter import format_single_result
from ocr_solver.models import DetailedSolverResult, QuestionOutput, SolveAttempt
from ocr_solver.noise.persian_noise import NoisePerturbation, PersianOCRNoiseInjector
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

# Custom High-Quality Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    
    .answer-banner {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    .answer-banner-fallback {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    .answer-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        opacity: 0.9;
    }
    .answer-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 2px;
    }
    
    .content-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
        direction: rtl;
        text-align: right;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #1E293B;
    }
    
    .noisy-box {
        background-color: #FFFBEB;
        border: 1.5px dashed #F59E0B;
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 14px;
        margin-bottom: 16px;
        direction: rtl;
        text-align: right;
        font-size: 1.02rem;
        line-height: 1.8;
        color: #78350F;
    }
    
    .reasoning-box {
        background-color: #F8FAFC;
        border-right: 4px solid #3B82F6;
        border-left: none;
        border-radius: 8px 0 0 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #1E293B;
        font-size: 0.98rem;
        line-height: 1.8;
        direction: rtl;
        text-align: right;
        font-family: Tahoma, sans-serif;
    }

    
    .pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-green { background-color: #DCFCE7; color: #15803D; }
    .pill-amber { background-color: #FEF3C7; color: #B45309; }
    .pill-blue { background-color: #DBEAFE; color: #1D4ED8; }
    .pill-purple { background-color: #F3E8FF; color: #7E22CE; }
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


def safe_update_status(status_container, **kwargs):
    """Safely update st.status container if available."""
    if status_container is not None and hasattr(status_container, "update"):
        try:
            status_container.update(**kwargs)
        except Exception:
            pass


def render_reasoning_markdown(reasoning_text: str) -> None:
    """
    Render LLM mathematical reasoning using st.markdown(..., unsafe_allow_html=True)
    with structural RTL styling and full LaTeX math rendering support.
    """
    if not reasoning_text:
        st.caption("No reasoning text available.")
        return

    formatted_html = (
        f'<div dir="rtl" style="text-align: right; line-height: 1.8; font-family: Tahoma, sans-serif;" class="reasoning-box">\n\n'
        f'{reasoning_text}\n\n'
        f'</div>'
    )
    st.markdown(formatted_html, unsafe_allow_html=True)


def execute_stepwise_solving(
    image_path: Path,
    inject_noise: bool = False,
    noise_rate: float = 0.08,
    max_perturbations: int = 3,
    max_retries: int = 3,
) -> Tuple[DetailedSolverResult, Optional[str], list[NoisePerturbation]]:
    """
    Executes the OCR-aware solver step-by-step, capturing full reasoning logs,
    and rendering each step live into Streamlit status & expandable containers.
    """
    ocr_client = DatalabOCRClient()
    noise_injector = PersianOCRNoiseInjector()
    agent = OCRAwareSolverAgent(max_retries=max_retries)

    corrupted_ocr_text: Optional[str] = None
    applied_perturbations: list[NoisePerturbation] = []

    with st.status("🧠 Agent is solving with OCR-Aware Refinement...", expanded=True) as status:
        # ---------------------------------------------------------
        # Step 1: Initial OCR Extraction
        # ---------------------------------------------------------
        st.write("#### 📄 Step 1: Initial OCR Text Extraction")
        ocr_res = ocr_client.extract_text(image_path)
        initial_text = ocr_res.text
        original_ocr = initial_text

        with st.expander("🔍 View Initial Extracted OCR Text", expanded=False):
            st.markdown(f'<div class="content-box">{initial_text}</div>', unsafe_allow_html=True)
            st.caption(f"Provider: **{ocr_res.provider}** | Status: **{'Success' if ocr_res.success else 'Failed'}**")

        # Step 1b: Synthetic Noise Injection (if requested)
        if inject_noise:
            st.write("#### ⚡ Step 1b: Synthetic Persian OCR Noise Perturbation")
            corrupted_text, perts = noise_injector.perturb_text(
                initial_text,
                char_rate=noise_rate,
                max_perturbations=max_perturbations,
            )
            corrupted_ocr_text = corrupted_text
            applied_perturbations = perts

            with st.expander(f"⚠️ Perturbations Applied ({len(perts)} character mutations)", expanded=True):
                for p in perts:
                    cat_name = p.category.value if hasattr(p.category, "value") else str(p.category)
                    st.markdown(
                        f"- Mutated character **`{p.original_char}`** $\\to$ **`{p.perturbed_char}`** "
                        f"*(Category: `{cat_name}`)* at offset `{p.index}`"
                    )
                st.markdown("**Scrambled OCR Text fed to Agent:**")
                st.markdown(f'<div class="noisy-box">{corrupted_text}</div>', unsafe_allow_html=True)

            initial_text = corrupted_text

        # ---------------------------------------------------------
        # Step 2: Attempt 1 (Initial LLM Solve)
        # ---------------------------------------------------------
        st.write("#### 📐 Step 2: Initial LLM Solving Attempt")
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

        with st.expander(f"📝 Attempt 1: Model Reasoning & Guessed Output", expanded=True):
            st.markdown(f"**Available Candidate Options:**")
            opt_pills = " ".join([f"<span class='pill pill-blue'>Option {k}: {v}</span>" for k, v in options.items()])
            st.markdown(opt_pills, unsafe_allow_html=True)

            st.markdown("**LLM Mathematical Reasoning:**")
            render_reasoning_markdown(attempt_result.reasoning)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Derived Computed Value", str(attempt_result.computed_value or "None"))
            with c2:
                guess_label = f"Option {attempt_result.matched_option}" if attempt_result.matched_option else "❌ No Match (Discrepancy)"
                st.metric("Initial Match Verdict", guess_label)

        # ---------------------------------------------------------
        # Step 3: Option Matching Verification
        # ---------------------------------------------------------
        st.write("#### 🎯 Step 3: Option Matching Verification")
        if attempt_result.matched_option:
            st.success(
                f"✅ **Immediate Match on Attempt 1!** Computed value `{attempt_result.computed_value}` cleanly matches "
                f"**Option {attempt_result.matched_option}** with confidence {attempt_result.match_confidence:.0%}."
            )
            safe_update_status(
                status,
                label=f"✅ Solved successfully on Attempt 1 (Answer: Option {attempt_result.matched_option})",
                state="complete",
                expanded=False,
            )
            return (
                DetailedSolverResult(
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
                ),
                corrupted_ocr_text,
                applied_perturbations,
            )

        st.warning(
            f"❌ **Option Mismatch Detected!** Derived computed value `{attempt_result.computed_value}` "
            f"does not match any of the candidate options `{list(options.keys())}`. "
            f"The agent treats this discrepancy as evidence of transcription errors and initiates visual refinement."
        )

        # ---------------------------------------------------------
        # Step 4: Visual OCR Refinement & Retry Loop
        # ---------------------------------------------------------
        while attempt_num <= agent.max_retries:
            st.write(f"#### 🔄 Step 4: Visual OCR Refinement & Re-solving (Iteration {attempt_num})")

            # 4a: Visual re-inspection of scan crop
            correction_res = agent._refine_ocr_with_vision(
                image_path=image_path,
                current_text=current_text,
                previous_attempt=attempts[-1],
            )
            corrected_text = correction_res.get("corrected_question_text") or current_text
            notes = correction_res.get("correction_notes", "")
            errors = correction_res.get("identified_errors", [])

            with st.expander(f"🔍 Visual Inspection & Error Diagnosis (Attempt {attempt_num})", expanded=True):
                st.markdown("**Identified Transcription Discrepancies:**")
                if errors:
                    for err in errors:
                        st.markdown(f"- ⚠️ **{err}**")
                else:
                    st.markdown("- *Minor structural or character transcription mismatch diagnosed.*")

                st.markdown(f"**Vision Model Diagnostic Notes:** {notes}")

                # Check if text was corrected
                if clean_text(corrected_text) != clean_text(current_text):
                    is_changed = True
                    current_text = corrected_text
                    options = extract_options_from_text(current_text)
                    st.info("💡 **Question text was corrected based on image crop re-inspection:**")
                    st.markdown(f'<div class="content-box">{current_text}</div>', unsafe_allow_html=True)
                else:
                    st.caption("ℹ️ OCR text was confirmed without textual changes.")

            attempt_num += 1

            # 4b: Re-solve with corrected text
            st.write(f"#### 📐 Re-solving Question with Corrected OCR (Attempt {attempt_num})")
            attempt_result = agent._execute_solve_attempt(
                image_path=image_path,
                question_text=current_text,
                attempt_number=attempt_num,
                is_correction=True,
                correction_notes=notes,
            )
            attempts.append(attempt_result)

            with st.expander(f"📝 Attempt {attempt_num}: Refined Reasoning & Output", expanded=True):
                st.markdown("**Updated Candidate Options:**")
                opt_pills = " ".join([f"<span class='pill pill-purple'>Option {k}: {v}</span>" for k, v in options.items()])
                st.markdown(opt_pills, unsafe_allow_html=True)

                st.markdown("**Mathematical Reasoning with Corrected Formulation:**")
                render_reasoning_markdown(attempt_result.reasoning)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("New Computed Value", str(attempt_result.computed_value or "None"))
                with c2:
                    new_verdict = f"Option {attempt_result.matched_option}" if attempt_result.matched_option else "❌ Still Mismatched"
                    st.metric("Option Match Result", new_verdict)

            # Check matching
            if attempt_result.matched_option:
                st.success(
                    f"🎉 **Refinement Succeeded on Attempt {attempt_num}!** "
                    f"Computed value `{attempt_result.computed_value}` matches **Option {attempt_result.matched_option}** "
                    f"(Confidence: {attempt_result.match_confidence:.0%})."
                )
                safe_update_status(
                    status,
                    label=f"✅ Resolved on Attempt {attempt_num} after visual refinement (Answer: Option {attempt_result.matched_option})",
                    state="complete",
                    expanded=False,
                )
                return (
                    DetailedSolverResult(
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
                    ),
                    corrupted_ocr_text,
                    applied_perturbations,
                )

        # ---------------------------------------------------------
        # Fallback if retry cap reached
        # ---------------------------------------------------------
        st.write("#### ⚖️ Step 5: Bounded Fallback Resolution")
        st.warning(f"Retry cap of {agent.max_retries} reached. Engaging best-guess heuristic resolution.")
        best_guess, fallback_reason = agent._resolve_fallback(
            image_path=image_path,
            question_text=current_text,
            options=options,
            attempts=attempts,
        )

        with st.expander("🛡️ Fallback Heuristic Justification", expanded=True):
            st.markdown(f"**Selected Best-Guess Candidate:** `Option {best_guess}`")
            st.markdown("**Heuristic Justification:**")
            render_reasoning_markdown(fallback_reason)

        safe_update_status(
            status,
            label=f"⚠️ Concluded with Best-Guess Option {best_guess} (Retry cap reached)",
            state="complete",
            expanded=False,
        )

        return (
            DetailedSolverResult(
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
            ),
            corrupted_ocr_text,
            applied_perturbations,
        )


def main():
    # Header
    st.markdown('<div class="main-title">🔍 OCR-Aware Question Solving Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive visualizer for self-correcting OCR refinement, mathematical reasoning, and option matching.</div>',
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
            help="Simulates realistic Persian character confusion and OCR degradation to test error recovery.",
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
    col_left, col_right = st.columns([1, 1.35], gap="large")

    # Placeholders for dynamic content
    with col_left:
        st.subheader("🖼️ Question Image")
        if image_path and image_path.exists():
            try:
                img = Image.open(image_path)
                st.image(img, caption=f"File: {image_path.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.info("👈 Please select or upload a question image from the sidebar to begin.")

        # Prominent Noisy Text display placeholder right below the image
        noisy_text_container = st.container()

    with col_right:
        st.subheader("🚀 Execution & Workflow Log")

        if image_path and image_path.exists():
            run_button = st.button("▶️ Solve Question", type="primary", use_container_width=True)

            if run_button:
                # Run the step-by-step solver
                result, corrupted_text, perts = execute_stepwise_solving(
                    image_path=image_path,
                    inject_noise=inject_noise,
                    noise_rate=noise_rate,
                    max_perturbations=max_noise_perturbations,
                    max_retries=max_retries,
                )

                # If noise was injected, populate the prominent corrupted text box below the image
                if inject_noise and corrupted_text:
                    with noisy_text_container:
                        st.markdown("### ⚠️ Corrupted OCR Text (Noise Injected)")
                        st.markdown(
                            f"Below is the degraded OCR text injected with **{len(perts)} synthetic perturbation(s)** "
                            f"before entering the agent's self-healing loop:"
                        )
                        st.markdown(f'<div class="noisy-box">{corrupted_text}</div>', unsafe_allow_html=True)

                st.divider()
                st.subheader("📊 Final Human-Readable Results")

                # Prominent Final Answer Banner
                if result.is_resolved:
                    st.markdown(
                        f"""
                        <div class="answer-banner">
                            <div class="answer-label">✅ Final Selected Option</div>
                            <div class="answer-value">Option {result.output.answer}</div>
                            <div style="margin-top: 6px; font-size: 0.92rem; opacity: 0.95;">
                                Resolved in <strong>{result.total_attempts}</strong> attempt(s) 
                                &bull; OCR Text Status: <strong>{'Corrected via Visual Refinement' if result.output.changed else 'Original OCR Preserved'}</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="answer-banner-fallback">
                            <div class="answer-label">⚠️ Fallback Selected Option (Cap Reached)</div>
                            <div class="answer-value">Option {result.output.answer}</div>
                            <div style="margin-top: 6px; font-size: 0.92rem; opacity: 0.95;">
                                Total Attempts: <strong>{result.total_attempts}</strong> &bull; Reason: {result.unresolved_reason or 'Heuristic fallback'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Clean, human-readable Final Question Text
                st.markdown("#### 📝 Final Question Text")
                st.markdown(f'<div class="content-box">{result.output.question_text}</div>', unsafe_allow_html=True)

                # Clean, human-readable Original OCR Text
                st.markdown("#### 📄 Original Extracted OCR Text")
                st.markdown(f'<div class="content-box">{result.output.original_ocr_text}</div>', unsafe_allow_html=True)

                # Summary Details & Download
                with st.expander("📦 Export / Audit Details", expanded=False):
                    final_json_dict = format_single_result(result)
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
