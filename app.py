import streamlit as st
from majorproject_1 import (
    analyze_problem,
    analyze_option,
    generate_arguments,
    run_debate,
    score_arguments,
    make_decision,
    generate_explanation
)
 
# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="DecisionAI", page_icon="🧠", layout="centered")
 
st.title("🧠 DecisionAI: Intelligent Reasoning System")
st.markdown("Make smarter decisions using AI-powered debate and reasoning — works for **any domain**: food, career, finance, education, tech, travel, and more.")
 
# ---------------- EXAMPLE HINTS ----------------
with st.expander("💡 Example problems you can try"):
    st.markdown("""
    - **Food:** *"I am hungry, what should I eat?"* → Dal Rice vs Burger  
    - **Career:** *"Which job should I take?"* → Startup vs Government Job  
    - **Finance:** *"Where should I invest my money?"* → Mutual Fund vs Fixed Deposit  
    - **Education:** *"Which course should I do?"* → MBA vs Online Certification  
    - **Tech:** *"Which field is better?"* → AI vs Robotics  
    - **Travel:** *"Where should I go for vacation?"* → Goa vs Manali  
    """)
 
# ---------------- INPUT SECTION ----------------
st.subheader("📥 Enter Your Decision Problem")
 
problem = st.text_input("Describe your problem or situation")
 
col1, col2 = st.columns(2)
with col1:
    opt1 = st.text_input("Option 1")
with col2:
    opt2 = st.text_input("Option 2")
 
# ---------------- PROCESS BUTTON ----------------
if st.button("⚖️ Analyze Decision", use_container_width=True):
 
    if not problem or not opt1 or not opt2:
        st.warning("⚠️ Please fill in all three fields before analyzing.")
    else:
        with st.spinner("Thinking... generating debate arguments..."):
 
            p  = problem.strip()
            o1 = opt1.strip()
            o2 = opt2.strip()
 
            keywords  = analyze_problem(p)
            features1 = analyze_option(o1)
            features2 = analyze_option(o2)
 
            args1_for, args1_against = generate_arguments(o1, keywords, features1)
            args2_for, args2_against = generate_arguments(o2, keywords, features2)
 
            debate      = run_debate(o1, o2, args1_for, args1_against, args2_for, args2_against)
            score1      = score_arguments(args1_for, args1_against, keywords, features1)
            score2      = score_arguments(args2_for, args2_against, keywords, features2)
            decision, confidence = make_decision(score1, score2, o1, o2)
            explanation = generate_explanation(decision, keywords, features1, features2, o1, o2)
 
        st.divider()
 
        # ── FINAL DECISION ────────────────────────────────
        st.subheader("🏆 Final Decision")
 
        if decision == o1:
            st.success(f"✅ **{o1}** is the better choice")
        elif decision == o2:
            st.success(f"✅ **{o2}** is the better choice")
        else:
            st.info("🤝 Both options are equally suitable")
 
        st.metric(label="Confidence Level", value=f"{confidence}%")
 
        # ── SCORE COMPARISON ─────────────────────────────
        st.subheader("📊 Score Comparison")
 
        if decision == o1:
            pct1 = confidence
            pct2 = 100 - confidence
        elif decision == o2:
            pct2 = confidence
            pct1 = 100 - confidence
        else:
            pct1 = pct2 = 50
 
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{o1}**")
            st.progress(pct1 / 100)
            st.caption(f"Raw score: {score1}  |  Relative: {pct1}%")
        with col_b:
            st.markdown(f"**{o2}**")
            st.progress(pct2 / 100)
            st.caption(f"Raw score: {score2}  |  Relative: {pct2}%")
 
        # ── DETECTED FEATURES ────────────────────────────
        st.subheader("🔍 Detected Features")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"**{o1}**")
            if features1:
                for f in features1:
                    st.markdown(f"- `{f}`")
            else:
                st.caption("No specific features detected")
        with fc2:
            st.markdown(f"**{o2}**")
            if features2:
                for f in features2:
                    st.markdown(f"- `{f}`")
            else:
                st.caption("No specific features detected")
 
        # ── EXPLANATION ──────────────────────────────────
        st.subheader("💡 Reasoning")
        st.info(explanation)
 
        # ── DEBATE DETAILS ───────────────────────────────
        with st.expander("⚖️ View Full Debate Process"):
            for line in debate:
                st.write(line)
 
        # ── DETECTED KEYWORDS ────────────────────────────
        with st.expander("🧠 Detected Keywords from Problem"):
            st.write(", ".join(keywords) if keywords else "None detected")