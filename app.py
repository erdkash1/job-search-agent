import streamlit as st
import os

st.set_page_config(  # ← must be FIRST streamlit command
    page_title="Job Search AI Agent",
    page_icon="🤖",
    layout="wide"
)

from agent import generate_cover_letter, JOBS, BLOCKED


st.write("ENV CHECK:", "GROQ_API_KEY" in os.environ)
st.write("KEY:", os.environ.get("GROQ_API_KEY", "NOT FOUND")[:8] if os.environ.get("GROQ_API_KEY") else "NOT FOUND")

st.title("🤖 Job Search AI Agent")
st.markdown("*Powered by LangChain + Groq (Llama 3)*")
st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Blocked Companies:**")
    for company in BLOCKED:
        st.markdown(f"🚫 {company.title()}")
    
    st.divider()
    st.markdown("**Candidate:**")
    st.markdown("👤 Erdenesuren Shirmen")
    st.markdown("🎓 Missouri State University CS 2026")
    st.markdown("🎯 QA Engineer / SDET / Java Dev")
    st.markdown("✅ OPT (STEM eligible)")

st.subheader("🔍 Search Jobs")
query = st.text_input(
    "Enter job title or skills:",
    placeholder="e.g. QA Automation Engineer Selenium Java"
)

col1, col2 = st.columns([1, 4])
with col1:
    search_button = st.button("🔍 Search", type="primary")

if search_button and query:
    with st.spinner("🤖 AI Agent searching and ranking jobs..."):
        
        from agent import search_jobs, filter_blocked, rank_jobs
        
        keywords = query.split()
        found_jobs = search_jobs(keywords)
        
        allowed_jobs, blocked_jobs = filter_blocked(found_jobs)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jobs Found", len(found_jobs))
        with col2:
            st.metric("After Filtering", len(allowed_jobs))
        with col3:
            st.metric("Blocked", len(blocked_jobs))
        
        if blocked_jobs:
            st.warning(f"🚫 Filtered out: {', '.join(j['company'] for j in blocked_jobs)}")
        
        if not allowed_jobs:
            st.error("No jobs found! Try different keywords.")
        else:
            with st.spinner("🤖 Ranking jobs by best fit..."):
                ranked_jobs = rank_jobs(allowed_jobs)
            
            st.divider()
            st.subheader(f"📋 Top {len(ranked_jobs)} Jobs For You")
            
            for i, job in enumerate(ranked_jobs, 1):
                with st.expander(
                    f"#{i} {'🥇' if i==1 else '🥈' if i==2 else '🥉' if i==3 else '▪️'} "
                    f"{job['title']} at {job['company']} — {job['location']}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Company:** {job['company']}")
                        st.markdown(f"**Location:** 📍 {job['location']}")
                        st.markdown(f"**Skills:** 🛠 {', '.join(job['skills'])}")
                        st.markdown(f"**URL:** 🔗 [{job['url']}]({job['url']})")
                    
                    with col2:
                        if st.button(f"✉️ Generate Cover Letter", key=f"cover_{i}"):
                            with st.spinner("Writing cover letter..."):
                                cover_letter = generate_cover_letter(job)
                            st.markdown("### ✉️ Cover Letter")
                            st.markdown(cover_letter)
                            st.download_button(
                                "📥 Download Cover Letter",
                                cover_letter,
                                file_name=f"cover_letter_{job['company']}.txt",
                                key=f"download_{i}"
                            )

elif search_button and not query:
    st.warning("Please enter a job title or skills to search!")

st.divider()
st.markdown(
    "Built with ❤️ by Erdenesuren Shirmen | "
    "[GitHub](https://github.com/erdkash1) | "
    "[LinkedIn](https://linkedin.com/in/erdenesuren-shirmen-dev)"
)