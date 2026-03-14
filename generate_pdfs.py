"""
generate_pdfs.py
----------------
Generates two professional hospital PDF documents:
  1. data/strategic_plan.pdf  — Nawaloka Hospital Negombo Sri Lanka Strategic Plan 2025-2028
  2. data/action_plan.pdf     — Nawaloka Hospital Negombo Sri Lanka Action Plan 2025
"""

import json
import os
from datetime import date
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus import PageBreak

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
SP_JSON    = os.path.join(DATA_DIR, "strategic_plan.json")
AP_JSON    = os.path.join(DATA_DIR, "action_plan.json")
SP_PDF     = os.path.join(DATA_DIR, "strategic_plan.pdf")
AP_PDF     = os.path.join(DATA_DIR, "action_plan.pdf")

# ── Hospital identity ──────────────────────────────────────────────────────────
HOSPITAL_NAME   = "Nawaloka Hospital Negombo"
HOSPITAL_TAG    = "Committed to Excellence in Healthcare"
ADDRESS_LINE    = "Negombo, Sri Lanka  |  www.nawaloka.lk  |  +94 31 222 4444"
STRATEGIC_TITLE = "Strategic Plan 2025 – 2028"
ACTION_TITLE    = "Operational Action Plan 2025"
APPROVED_BY     = "Board of Directors — Nawaloka Hospital Negombo"
VERSION_SP      = "Version 3.1  |  Approved January 2025"
VERSION_AP      = "Version 2.4  |  Approved February 2025"

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#003366")
TEAL       = colors.HexColor("#006B8F")
LIGHT_TEAL = colors.HexColor("#E0F0F5")
GOLD       = colors.HexColor("#B8860B")
LIGHT_GREY = colors.HexColor("#F4F6F8")
MID_GREY   = colors.HexColor("#BDC3C7")
DARK_GREY  = colors.HexColor("#4A4A4A")
WHITE      = colors.white

# Objective accent colours (one per objective)
OBJ_COLOURS = [
    colors.HexColor("#006B8F"),   # O1 teal
    colors.HexColor("#1A6B3A"),   # O2 green
    colors.HexColor("#7B3FA0"),   # O3 purple
    colors.HexColor("#C0522A"),   # O4 rust
    colors.HexColor("#B8860B"),   # O5 gold
]


# ══════════════════════════════════════════════════════════════════════════════
# Style helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_styles():
    base = getSampleStyleSheet()
    s = {}

    s["hospital_name"] = ParagraphStyle(
        "hospital_name", fontName="Helvetica-Bold",
        fontSize=22, textColor=WHITE, alignment=TA_LEFT, leading=26
    )
    s["hospital_tag"] = ParagraphStyle(
        "hospital_tag", fontName="Helvetica-Oblique",
        fontSize=10, textColor=colors.HexColor("#CCE8F0"), alignment=TA_LEFT
    )
    s["doc_title"] = ParagraphStyle(
        "doc_title", fontName="Helvetica-Bold",
        fontSize=28, textColor=WHITE, alignment=TA_LEFT, spaceBefore=16, leading=34
    )
    s["cover_meta"] = ParagraphStyle(
        "cover_meta", fontName="Helvetica",
        fontSize=9, textColor=colors.HexColor("#CCE8F0"), alignment=TA_LEFT, leading=14
    )
    s["address"] = ParagraphStyle(
        "address", fontName="Helvetica",
        fontSize=8, textColor=MID_GREY, alignment=TA_CENTER
    )
    s["section_heading"] = ParagraphStyle(
        "section_heading", fontName="Helvetica-Bold",
        fontSize=14, textColor=NAVY, spaceBefore=18, spaceAfter=4, leading=18
    )
    s["obj_title"] = ParagraphStyle(
        "obj_title", fontName="Helvetica-Bold",
        fontSize=12, textColor=WHITE, leading=15
    )
    s["obj_body"] = ParagraphStyle(
        "obj_body", fontName="Helvetica",
        fontSize=9.5, textColor=DARK_GREY, leading=14, alignment=TA_JUSTIFY
    )
    s["body"] = ParagraphStyle(
        "body", fontName="Helvetica",
        fontSize=9.5, textColor=DARK_GREY, leading=14, alignment=TA_JUSTIFY,
        spaceBefore=4, spaceAfter=4
    )
    s["body_bold"] = ParagraphStyle(
        "body_bold", fontName="Helvetica-Bold",
        fontSize=9.5, textColor=DARK_GREY, leading=14
    )
    s["small_italic"] = ParagraphStyle(
        "small_italic", fontName="Helvetica-Oblique",
        fontSize=8, textColor=MID_GREY, leading=11
    )
    s["toc_heading"] = ParagraphStyle(
        "toc_heading", fontName="Helvetica-Bold",
        fontSize=12, textColor=NAVY, spaceBefore=6, spaceAfter=2
    )
    s["toc_entry"] = ParagraphStyle(
        "toc_entry", fontName="Helvetica",
        fontSize=10, textColor=DARK_GREY, leading=16, leftIndent=12
    )
    s["action_id"] = ParagraphStyle(
        "action_id", fontName="Helvetica-Bold",
        fontSize=11, textColor=WHITE, leading=13, alignment=TA_CENTER
    )
    s["action_title"] = ParagraphStyle(
        "action_title", fontName="Helvetica-Bold",
        fontSize=10.5, textColor=NAVY, leading=14
    )
    s["action_body"] = ParagraphStyle(
        "action_body", fontName="Helvetica",
        fontSize=9, textColor=DARK_GREY, leading=13, alignment=TA_JUSTIFY
    )
    s["tag_text"] = ParagraphStyle(
        "tag_text", fontName="Helvetica-Bold",
        fontSize=8, textColor=TEAL, leading=10
    )
    s["footer_text"] = ParagraphStyle(
        "footer_text", fontName="Helvetica",
        fontSize=7.5, textColor=MID_GREY, alignment=TA_CENTER
    )
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Page templates (header / footer callbacks)
# ══════════════════════════════════════════════════════════════════════════════

class HospitalDoc(BaseDocTemplate):
    """Custom doc template that draws a running header and footer on every page
    except the cover page (page 1)."""

    def __init__(self, filename, doc_subtitle, **kwargs):
        super().__init__(filename, **kwargs)
        self.doc_subtitle = doc_subtitle
        self._styles = build_styles()

    def handle_pageBegin(self):
        self._handle_pageBegin()

    def afterPage(self):
        canvas = self.canv
        page_num = canvas.getPageNumber()
        w, h = A4

        if page_num == 1:
            return  # cover page — no running header

        # ── Running header ─────────────────────────────────────────────────
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 1.4*cm, w, 1.4*cm, fill=1, stroke=0)

        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(WHITE)
        canvas.drawString(1.5*cm, h - 0.9*cm, HOSPITAL_NAME)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#CCE8F0"))
        right_x = w - 1.5*cm
        canvas.drawRightString(right_x, h - 0.9*cm, self.doc_subtitle)

        # ── Footer ─────────────────────────────────────────────────────────
        canvas.setFillColor(LIGHT_GREY)
        canvas.rect(0, 0, w, 1.1*cm, fill=1, stroke=0)
        canvas.setStrokeColor(MID_GREY)
        canvas.setLineWidth(0.5)
        canvas.line(1.5*cm, 1.1*cm, w - 1.5*cm, 1.1*cm)

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MID_GREY)
        canvas.drawCentredString(w / 2, 0.45*cm,
            f"{HOSPITAL_NAME}  |  {self.doc_subtitle}  |  Page {page_num}")
        canvas.restoreState()


def make_doc(path, subtitle):
    margin = 1.8*cm
    doc = HospitalDoc(
        path,
        doc_subtitle=subtitle,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.2*cm, bottomMargin=1.8*cm,
    )
    frame = Frame(
        margin, 1.8*cm,
        A4[0] - 2*margin, A4[1] - 2.2*cm - 1.8*cm,
        id="main"
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# Shared cover builder
# ══════════════════════════════════════════════════════════════════════════════

def cover_page(s, title, version, description_paragraphs):
    """Returns a list of flowables that form the cover page."""
    w, h = A4
    story = []

    # Full-bleed navy header block — implemented as a wide table row
    cover_data = [[
        Paragraph(f"{HOSPITAL_NAME}", s["hospital_name"]),
    ]]
    cover_table = Table(cover_data, colWidths=[w - 3.6*cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
    ]))
    story.append(cover_table)

    # Tagline bar
    tag_data = [[Paragraph(HOSPITAL_TAG, s["hospital_tag"])]]
    tag_table = Table(tag_data, colWidths=[w - 3.6*cm])
    tag_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
    ]))
    story.append(tag_table)
    story.append(Spacer(1, 1.6*cm))

    # Document title
    title_data = [[Paragraph(title, s["doc_title"])]]
    title_table = Table(title_data, colWidths=[w - 3.6*cm])
    title_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 22),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 22),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 0.6*cm))

    # Meta block
    meta_rows = [
        [Paragraph("<b>Approved by:</b>",  s["cover_meta"]),
         Paragraph(APPROVED_BY,            s["cover_meta"])],
        [Paragraph("<b>Date:</b>",         s["cover_meta"]),
         Paragraph(f"January 2025",        s["cover_meta"])],
        [Paragraph("<b>Document ref:</b>", s["cover_meta"]),
         Paragraph(version,                s["cover_meta"])],
        [Paragraph("<b>Classification:</b>", s["cover_meta"]),
         Paragraph("INTERNAL – NOT FOR PUBLIC DISTRIBUTION", s["cover_meta"])],
    ]
    meta_table = Table(meta_rows, colWidths=[3.5*cm, (w - 3.6*cm) - 3.5*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 1*cm))

    # Horizontal rule
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL))
    story.append(Spacer(1, 0.5*cm))

    # Executive summary / intro
    for para in description_paragraphs:
        story.append(Paragraph(para, s["body"]))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 1*cm))

    # Footer address line
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(ADDRESS_LINE, s["address"]))

    story.append(PageBreak())
    return story


# ══════════════════════════════════════════════════════════════════════════════
# Per-objective rich content (rationale, targets, risks, timeline, actions)
# ══════════════════════════════════════════════════════════════════════════════

OBJ_DETAIL = {
    "O1": {
        "owner":   "Dr. Marcus Osei, Director of Quality & Patient Safety",
        "rationale": (
            "Patient safety is the cornerstone of clinical excellence. Nawaloka Hospital Negombo "
            "currently records a hospital-acquired infection (HAI) rate of 3.1% — above the NHS "
            "England benchmark of 2.0%. In 2024, 47 serious clinical incidents were reported, of "
            "which 12 were classified as Never Events. Medication errors accounted for 31% of all "
            "adverse incident reports. The Trust Board has made patient safety the highest-priority "
            "objective for 2025–2028, with a zero-tolerance approach to preventable patient harm."
            "<br/><br/>"
            "Achieving this objective requires a system-wide cultural shift — from reactive incident "
            "management to proactive clinical risk prevention. This will be driven by mandatory staff "
            "training, rigorous monthly audits, a state-of-the-art digital incident reporting platform, "
            "and the attainment of Joint Commission International (JCI) accreditation, which will "
            "independently certify the Trust's safety standards against global best practice."
        ),
        "targets": [
            ["Key Performance Indicator",         "Baseline (2024)", "Target (2028)"],
            ["Hospital-acquired infection rate",  "3.1%",            "< 2.0%"],
            ["Serious clinical incidents (p.a.)", "47",              "< 20"],
            ["Never Events reported",             "12",              "0"],
            ["Medication error rate (per 1,000)", "4.2",             "< 1.5"],
            ["JCI accreditation status",          "Not held",        "Certified"],
            ["Staff safety training completion",  "62%",             "100%"],
        ],
        "risks": [
            ["Risk", "Likelihood", "Mitigation"],
            ["Staff resistance to new reporting system",     "Medium", "Change management programme; champion network"],
            ["JCI accreditation delayed by resource gaps",   "Medium", "Dedicated project team; external consultancy support"],
            ["HAI outbreak during audit cycle",              "Low",    "Rapid response protocol; surge isolation capacity"],
            ["High agency staff unfamiliar with protocols",  "High",   "Mandatory induction module for all bank/agency staff"],
        ],
        "timeline": [
            ["Milestone",                                  "Target Date"],
            ["Incident reporting system go-live",          "March 2025"],
            ["First monthly infection audit cycle complete","April 2025"],
            ["Safety Training Programme cohort 1 complete","June 2025"],
            ["JCI pre-assessment audit",                   "September 2025"],
            ["JCI full accreditation submission",          "December 2025"],
            ["HAI rate below 2.5% sustained",              "June 2026"],
            ["HAI rate below 2.0% target met",             "December 2027"],
        ],
        "actions": ["A1 — Monthly Infection Control Audit",
                    "A2 — Staff Safety Training Programme",
                    "A3 — JCI Accreditation Preparation",
                    "A4 — Clinical Incident Reporting System"],
    },

    "O2": {
        "owner":   "Ms. Leila Moussaoui, Chief Information Officer",
        "rationale": (
            "Nawaloka Hospital Negombo's current IT infrastructure was last comprehensively "
            "upgraded in 2016. Paper-based clinical records remain in use across 8 of 14 wards. "
            "A 2024 infrastructure audit identified 23 critical cybersecurity vulnerabilities, "
            "and the Trust experienced one ransomware incident in 2023 that disrupted outpatient "
            "services for 11 days. Administrative staff spend an estimated 34% of their working "
            "time on manual data entry tasks that could be automated."
            "<br/><br/>"
            "This objective will transform the Trust into a digitally mature, data-driven "
            "organisation. The deployment of a cloud-hosted Electronic Health Record system will "
            "create a single source of truth for all patient data. AI-powered diagnostic tools "
            "will augment clinical decision-making. Automated scheduling and billing will recover "
            "thousands of staff hours annually, and a hardened cybersecurity posture will protect "
            "sensitive patient data in line with UK GDPR and NHS Data Security Standards."
        ),
        "targets": [
            ["Key Performance Indicator",               "Baseline (2024)", "Target (2028)"],
            ["Wards using paper records",               "8 of 14",         "0 of 14"],
            ["Critical cybersecurity vulnerabilities",  "23",              "0"],
            ["Admin time spent on manual data entry",   "34%",             "< 10%"],
            ["Appointment no-show rate",                "18%",             "< 12%"],
            ["AI diagnostic tool adoption (radiology)", "0%",              "> 80%"],
            ["Unplanned IT downtime (hours/year)",      "142 hrs",         "< 24 hrs"],
        ],
        "risks": [
            ["Risk", "Likelihood", "Mitigation"],
            ["EHR go-live delayed by integration issues",    "High",   "Phased rollout with parallel running period"],
            ["Staff resistance to digital workflows",        "Medium", "Superuser programme; hands-on training labs"],
            ["Cybersecurity breach during transition",       "Medium", "Penetration testing before each go-live phase"],
            ["AI diagnostic tool regulatory approval delay", "Low",    "Early engagement with MHRA and NHS AI Lab"],
        ],
        "timeline": [
            ["Milestone",                                    "Target Date"],
            ["Cybersecurity audit and remediation complete", "March 2025"],
            ["EHR pilot on 2 wards",                         "April 2025"],
            ["Automated scheduling system live",             "June 2025"],
            ["EHR full go-live across all wards",            "September 2025"],
            ["Patient self-service portal launched",         "October 2025"],
            ["AI diagnostic pilot (radiology)",              "November 2025"],
            ["Zero paper records across all wards",          "June 2026"],
        ],
        "actions": ["A5 — Deploy Electronic Health Record System",
                    "A6 — Automate Patient Scheduling and Billing",
                    "A7 — Upgrade Cybersecurity Infrastructure",
                    "A8 — Implement AI-Powered Diagnostic Support Tools"],
    },

    "O3": {
        "owner":   "Ms. Priya Chandran, Director of Patient Experience",
        "rationale": (
            "The Friends and Family Test (FFT) score for Nawaloka Hospital Negombo stood at 71% "
            "in Q4 2024, below the London average of 78%. Outpatient waiting times average 47 "
            "minutes beyond appointment time. Only 38% of patients rated their overall care "
            "experience as 'Excellent'. The Trust serves a catchment area with significant "
            "linguistic diversity — 31% of registered patients speak English as a second language "
            "— yet only basic translation services are currently available."
            "<br/><br/>"
            "This objective places the patient at the centre of every service decision. It will "
            "redesign the physical and digital patient journey from first contact to post-discharge. "
            "By introducing multilingual communication, dramatically reducing waiting times, "
            "deploying a self-service patient portal, and systematically collecting and acting on "
            "patient feedback, the Trust will achieve a step-change in satisfaction and loyalty."
        ),
        "targets": [
            ["Key Performance Indicator",               "Baseline (2024)", "Target (2028)"],
            ["Friends & Family Test (Excellent)",       "38%",             "> 65%"],
            ["Overall FFT positive score",              "71%",             "> 85%"],
            ["Average outpatient wait beyond appt time","47 mins",         "< 20 mins"],
            ["Languages with translation support",      "3",               "12"],
            ["Patient portal adoption rate",            "0%",              "> 60%"],
            ["Post-discharge survey response rate",     "12%",             "> 35%"],
        ],
        "risks": [
            ["Risk", "Likelihood", "Mitigation"],
            ["Survey fatigue reduces response rate",         "Medium", "Short digital surveys; incentive programme"],
            ["Portal adoption low among elderly patients",   "High",   "Assisted digital support service; paper fallback"],
            ["Interpreter availability gaps",                "Medium", "Video remote interpretation as backup service"],
            ["Wait time reduction targets not met",          "Medium", "Lean process review; additional clinic capacity"],
        ],
        "timeline": [
            ["Milestone",                                    "Target Date"],
            ["Patient satisfaction survey system launched",  "February 2025"],
            ["Multilingual interpreter service live (6 langs)","March 2025"],
            ["Outpatient triage redesign complete",          "May 2025"],
            ["Waiting area refurbishment completed",         "June 2025"],
            ["Patient self-service portal launched",         "October 2025"],
            ["FFT positive score exceeds 80%",               "December 2026"],
            ["All 12 languages covered",                     "March 2026"],
        ],
        "actions": ["A9  — Launch Patient Satisfaction Survey Programme",
                    "A10 — Reduce Outpatient Waiting Times Initiative",
                    "A11 — Introduce Multilingual Patient Communication Services",
                    "A12 — Deploy Patient Self-Service Portal"],
    },

    "O4": {
        "owner":   "Ms. Ama Forson, Director of People & Organisational Development",
        "rationale": (
            "The Trust's nursing vacancy rate stands at 14.2% — equivalent to 187 unfilled posts. "
            "Annual nursing turnover is 22%, well above the NHS England average of 15.8%. Exit "
            "interview data shows that 44% of departing nurses cite limited career development "
            "opportunities as a primary reason for leaving. Staff engagement scores rank Nawaloka Hospital "
            "in the bottom quartile of London trusts for the second consecutive year."
            "<br/><br/>"
            "A high-performing, motivated workforce is the single most important determinant of "
            "clinical quality. This objective will invest in our people through structured "
            "leadership development, evidence-based wellbeing interventions, competitive retention "
            "incentives, and formal university partnerships. The goal is to make Nawaloka Hospital "
            "Centre an employer of choice — attracting the best talent and keeping it."
        ),
        "targets": [
            ["Key Performance Indicator",               "Baseline (2024)", "Target (2028)"],
            ["Nursing vacancy rate",                    "14.2%",           "< 7%"],
            ["Annual nursing turnover",                 "22%",             "< 12%"],
            ["Staff engagement score (NHS survey)",     "Bottom quartile", "Top quartile"],
            ["Staff reporting burnout (quarterly)",     "38%",             "< 15%"],
            ["Leaders completing development programme","0",               "30 per year"],
            ["University internship places (p.a.)",     "12",              "50"],
        ],
        "risks": [
            ["Risk", "Likelihood", "Mitigation"],
            ["Competitive market makes retention hard",      "High",   "Benchmark pay regularly; non-pay benefits package"],
            ["Leadership programme uptake low",              "Medium", "Protected study time; line manager endorsement"],
            ["University partnership negotiations stall",    "Low",    "Engage NHS England workforce team for facilitation"],
            ["Wellbeing programme insufficient for burnout", "Medium", "Clinical psychology input; monitored outcomes"],
        ],
        "timeline": [
            ["Milestone",                                      "Target Date"],
            ["Nurse retention incentive package launched",     "January 2025"],
            ["Staff wellbeing programme live",                 "February 2025"],
            ["Leadership development cohort 1 enrolled",       "March 2025"],
            ["University partnership agreements signed",       "June 2025"],
            ["First internship cohort (50 students) starts",   "September 2025"],
            ["Nursing turnover below 18%",                     "December 2026"],
            ["Nursing vacancy rate below 10%",                 "December 2027"],
        ],
        "actions": ["A13 — Launch Clinical Leadership Development Programme",
                    "A14 — Staff Wellbeing and Mental Health Support Initiative",
                    "A15 — Nurse Retention and Incentive Scheme",
                    "A16 — University Medical Internship Partnership Programme"],
    },

    "O5": {
        "owner":   "Mr. David Reeves, Chief Financial Officer",
        "rationale": (
            "Nawaloka Hospital Negombo carried a LKR 320 million deficit at the close of the "
            "2023/24 financial year, against a planned break-even position. Non-clinical "
            "overhead costs grew by 11% in real terms over the past three years. Medical "
            "supply expenditure is 18% above the NHS national procurement benchmark due to "
            "fragmented purchasing across directorates. The Trust has not submitted a "
            "competitive grant application since 2021, leaving an estimated £1.2m per year "
            "of available public funding unclaimed."
            "<br/><br/>"
            "This objective will restore and sustain financial health without compromising "
            "clinical services. The approach combines rigorous cost discipline — through "
            "annual budget audits and supply chain optimisation — with active revenue "
            "diversification through private patient services and external grant funding. "
            "Every efficiency saving will be reinvested into frontline patient care."
        ),
        "targets": [
            ["Key Performance Indicator",               "Baseline (2024)", "Target (2028)"],
            ["Annual financial surplus / (deficit)",    "(£3.2m)",         "Surplus ≥ £1.0m"],
            ["Non-clinical overhead growth (real)",     "+11% (3yr)",      "< 0% (flat)"],
            ["Supply chain cost vs NHS benchmark",      "+18%",            "At or below benchmark"],
            ["Private patient revenue (p.a.)",          "£0.4m",           "≥ £2.4m"],
            ["Grant and subsidy income secured (p.a.)", "£0.1m",           "≥ £1.5m"],
            ["Cost improvement programme (CIP) delivery","62%",            "> 90%"],
        ],
        "risks": [
            ["Risk", "Likelihood", "Mitigation"],
            ["Private patient wing construction overrun",    "Medium", "Fixed-price contractor contract; contingency budget"],
            ["Grant applications unsuccessful",              "Medium", "Diversify funding portfolio; bid-writing expertise"],
            ["Supply chain savings not fully realised",      "Medium", "Centralised procurement governance board"],
            ["CIP schemes clinically undeliverable",         "Low",    "Clinical sign-off required on all CIP proposals"],
        ],
        "timeline": [
            ["Milestone",                                     "Target Date"],
            ["Annual budget review 2025/26 complete",         "February 2025"],
            ["Supply chain consolidation contracts signed",   "May 2025"],
            ["First grant applications submitted",            "June 2025"],
            ["Private patient wing planning approved",        "July 2025"],
            ["Private patient unit opens",                    "October 2025"],
            ["Supply cost benchmark achieved",                "March 2026"],
            ["Trust returns to financial surplus",            "March 2027"],
        ],
        "actions": ["A17 — Annual Budget Review and Cost Reduction Audit",
                    "A18 — Expand Private Patient and Revenue Services",
                    "A19 — Apply for Healthcare Grants and Government Subsidies",
                    "A20 — Medical Supply Chain Optimisation Programme"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIC PLAN PDF
# ══════════════════════════════════════════════════════════════════════════════

def _obj_page(s, obj, detail, index):
    """Build one full page of content for a single strategic objective."""
    colour  = OBJ_COLOURS[index % len(OBJ_COLOURS)]
    w       = A4[0]
    content_w = w - 3.6*cm

    blocks = []

    # ── Big coloured header banner ─────────────────────────────────────────
    banner_style = ParagraphStyle(
        "banner", fontName="Helvetica-Bold", fontSize=18,
        textColor=WHITE, leading=22
    )
    sub_style = ParagraphStyle(
        "sub", fontName="Helvetica-Oblique", fontSize=9.5,
        textColor=colors.HexColor("#DDEEF5"), leading=13
    )
    banner_data = [[
        Paragraph(f"{obj['id']}  |  {obj['title']}", banner_style),
        Paragraph(f"Owner: {detail['owner']}", sub_style),
    ]]
    banner = Table(banner_data, colWidths=[content_w * 0.62, content_w * 0.38])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colour),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
    ]))
    blocks.append(banner)
    blocks.append(Spacer(1, 0.35*cm))

    # ── Strategic Rationale ────────────────────────────────────────────────
    blocks.append(Paragraph("Strategic Rationale", s["section_heading"]))
    blocks.append(HRFlowable(width="100%", thickness=1, color=colour, spaceAfter=6))
    blocks.append(Paragraph(detail["rationale"], s["body"]))
    blocks.append(Spacer(1, 0.4*cm))

    # ── Two-column layout: Targets left, Supporting Actions right ──────────
    # Build targets table
    tgt_rows = detail["targets"]
    tgt_style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  colour),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica"),
    ]
    tgt_para_rows = []
    for ri, row in enumerate(tgt_rows):
        font = "Helvetica-Bold" if ri == 0 else "Helvetica"
        fc   = WHITE if ri == 0 else DARK_GREY
        tgt_para_rows.append([
            Paragraph(str(row[0]), ParagraphStyle("tc", fontName=font, fontSize=8, textColor=fc, leading=10)),
            Paragraph(str(row[1]), ParagraphStyle("tc", fontName=font, fontSize=8, textColor=fc, leading=10, alignment=TA_CENTER)),
            Paragraph(str(row[2]), ParagraphStyle("tc", fontName="Helvetica-Bold" if ri==0 else "Helvetica",
                                                  fontSize=8, textColor=colors.HexColor("#1A6B3A") if ri > 0 else WHITE,
                                                  leading=10, alignment=TA_CENTER)),
        ])
    col_left = content_w * 0.57
    tgt_table = Table(tgt_para_rows, colWidths=[col_left*0.56, col_left*0.22, col_left*0.22])
    tgt_table.setStyle(TableStyle(tgt_style))

    # Build supporting actions list
    action_label_style = ParagraphStyle(
        "al", fontName="Helvetica", fontSize=8.5, textColor=DARK_GREY, leading=13,
        leftIndent=8, bulletIndent=0
    )
    action_header = Table(
        [[Paragraph("Supporting Actions", ParagraphStyle(
            "ah", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE))]],
        colWidths=[content_w * 0.38]
    )
    action_header.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colour),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    action_rows = [[action_header]]
    for act in detail["actions"]:
        cell = Table(
            [[Paragraph(f"▸  {act}", action_label_style)]],
            colWidths=[content_w * 0.38]
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.4, MID_GREY),
        ]))
        action_rows.append([cell])
    actions_col = Table(action_rows, colWidths=[content_w * 0.38])
    actions_col.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))

    two_col = Table(
        [[tgt_table, actions_col]],
        colWidths=[content_w * 0.59, content_w * 0.41]
    )
    two_col.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (0, 0),   6),
    ]))

    blocks.append(Paragraph("Performance Targets", s["section_heading"]))
    blocks.append(HRFlowable(width="100%", thickness=1, color=colour, spaceAfter=6))
    blocks.append(two_col)
    blocks.append(Spacer(1, 0.4*cm))

    # ── Risks & Mitigations ────────────────────────────────────────────────
    blocks.append(Paragraph("Key Risks and Mitigations", s["section_heading"]))
    blocks.append(HRFlowable(width="100%", thickness=1, color=colour, spaceAfter=6))

    risk_rows = detail["risks"]
    risk_para_rows = []
    for ri, row in enumerate(risk_rows):
        font = "Helvetica-Bold" if ri == 0 else "Helvetica"
        fc   = WHITE if ri == 0 else DARK_GREY
        # Likelihood colour coding
        like_colour = fc
        if ri > 0:
            lv = str(row[1]).strip()
            like_colour = (colors.HexColor("#C0522A") if lv == "High"
                           else colors.HexColor("#B8860B") if lv == "Medium"
                           else colors.HexColor("#1A6B3A"))
        risk_para_rows.append([
            Paragraph(str(row[0]), ParagraphStyle("rc", fontName=font, fontSize=8, textColor=fc, leading=10)),
            Paragraph(str(row[1]), ParagraphStyle("rc", fontName="Helvetica-Bold" if ri>0 else font,
                                                  fontSize=8, textColor=like_colour, leading=10, alignment=TA_CENTER)),
            Paragraph(str(row[2]), ParagraphStyle("rc", fontName=font, fontSize=8, textColor=fc, leading=10)),
        ])
    risk_table = Table(risk_para_rows, colWidths=[content_w*0.42, content_w*0.12, content_w*0.46])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colour),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    blocks.append(risk_table)
    blocks.append(Spacer(1, 0.4*cm))

    # ── Delivery Timeline ──────────────────────────────────────────────────
    blocks.append(Paragraph("Delivery Timeline", s["section_heading"]))
    blocks.append(HRFlowable(width="100%", thickness=1, color=colour, spaceAfter=6))

    tl_rows = detail["timeline"]
    tl_para_rows = []
    for ri, row in enumerate(tl_rows):
        font = "Helvetica-Bold" if ri == 0 else "Helvetica"
        fc   = WHITE if ri == 0 else DARK_GREY
        tl_para_rows.append([
            Paragraph(str(row[0]), ParagraphStyle("tl", fontName=font, fontSize=8, textColor=fc, leading=10)),
            Paragraph(str(row[1]), ParagraphStyle("tl", fontName=font, fontSize=8, textColor=fc, leading=10, alignment=TA_CENTER)),
        ])
    tl_table = Table(tl_para_rows, colWidths=[content_w*0.70, content_w*0.30])
    tl_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colour),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    blocks.append(tl_table)

    blocks.append(PageBreak())
    return blocks


def build_strategic_plan_pdf():
    with open(SP_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    objectives = data["objectives"]

    s = build_styles()
    doc = make_doc(SP_PDF, STRATEGIC_TITLE)
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    intro = [
        "This Strategic Plan sets out the Board-approved direction for Nawaloka Hospital "
        "Centre over the three-year period 2025 to 2028. It defines five strategic "
        "objectives that will guide investment decisions, operational priorities, and "
        "performance accountability across all clinical and corporate directorates.",

        "The plan was developed through an extensive engagement process involving clinical "
        "leaders, frontline staff, patient representatives, commissioners, and the Board of "
        "Directors. It reflects our ambitions to deliver safe, high-quality, compassionate "
        "care while building organisational resilience and long-term sustainability.",

        "Each objective is presented on its own dedicated page, covering: strategic "
        "rationale and context, measurable performance targets, key risks and mitigations, "
        "a delivery timeline, and the specific operational actions that will drive delivery.",

        "Progress will be monitored through quarterly Board performance reports, with a "
        "formal mid-point review scheduled for January 2027.",
    ]
    story += cover_page(s, STRATEGIC_TITLE, VERSION_SP, intro)

    # ── Table of Contents ──────────────────────────────────────────────────
    story.append(Paragraph("Contents", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=6))

    toc_items = [
        ("1.", "Foreword from the Chief Executive"),
        ("2.", "Our Vision, Mission and Values"),
        ("3.", "Strategic Context"),
        ("4.", "Our Five Strategic Objectives (overview)"),
        ("5.", "Objective Pages — detailed one page per objective"),
    ]
    for num, text in toc_items:
        row = [[Paragraph(num, s["toc_entry"]), Paragraph(text, s["toc_entry"])]]
        t = Table(row, colWidths=[1*cm, 13*cm])
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ]))
        story.append(t)

    for i, obj in enumerate(objectives, 1):
        colour = OBJ_COLOURS[i-1 % len(OBJ_COLOURS)]
        row = [[
            Paragraph(f"5.{i}.", s["toc_entry"]),
            Paragraph(f"{obj['id']} — {obj['title']}", s["toc_entry"]),
        ]]
        t = Table(row, colWidths=[1.2*cm, 13*cm])
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
        ]))
        story.append(t)

    story.append(PageBreak())

    # ── Section 1 — Foreword ───────────────────────────────────────────────
    story.append(Paragraph("1.  Foreword from the Chief Executive", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))
    foreword = (
        "Nawaloka Hospital Negombo has served the communities of Negombo and the Western Province of Sri Lanka "
        "for over 80 years. Our dedicated teams deliver more than 320,000 patient episodes "
        "each year — and every one of those encounters is an opportunity to demonstrate our "
        "values of safety, compassion, innovation, and integrity."
        "<br/><br/>"
        "This Strategic Plan reflects the next chapter in our journey. It is ambitious and "
        "deliberate. We are setting ourselves clear, measurable objectives across patient "
        "safety, technology, patient experience, workforce development, and financial "
        "sustainability because we know that excellence in healthcare depends on all five "
        "dimensions working in harmony."
        "<br/><br/>"
        "I am proud to present this plan and confident that the entire Nawaloka team — "
        "clinical and non-clinical alike — will rise to the challenge it sets. The pages "
        "that follow set out not just what we want to achieve, but precisely how we will "
        "get there, who is accountable, and how we will know when we have succeeded."
        "<br/><br/>"
        "<i>Dr. Patricia Okonkwo, Chief Executive Officer<br/>"
        "Nawaloka Hospital Negombo, January 2025</i>"
    )
    story.append(Paragraph(foreword, s["body"]))
    story.append(Spacer(1, 0.8*cm))

    # ── Section 2 — Vision & Mission ──────────────────────────────────────
    story.append(Paragraph("2.  Our Vision, Mission and Values", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))
    vm_data = [
        [Paragraph("<b>Vision</b>",  s["body_bold"]),
         Paragraph("To be London's most trusted centre for safe, innovative, and compassionate healthcare.", s["body"])],
        [Paragraph("<b>Mission</b>", s["body_bold"]),
         Paragraph("To deliver outstanding clinical care, develop our people, harness technology, and ensure the long-term sustainability of services for every patient we serve.", s["body"])],
        [Paragraph("<b>Values</b>",  s["body_bold"]),
         Paragraph("Safety  ·  Compassion  ·  Integrity  ·  Innovation  ·  Inclusion", s["body"])],
    ]
    vm_table = Table(vm_data, colWidths=[2.8*cm, 13*cm])
    vm_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT_TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.5, MID_GREY),
    ]))
    story.append(vm_table)
    story.append(Spacer(1, 0.7*cm))

    # ── Section 3 — Strategic Context ─────────────────────────────────────
    story.append(Paragraph("3.  Strategic Context", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))
    context = (
        "The NHS faces an unprecedented combination of pressures: rising patient demand, "
        "workforce shortages, accelerating technological change, and the need to restore "
        "financial balance. Nawaloka Hospital Negombo is not immune to these challenges — "
        "but it is well positioned to address them."
        "<br/><br/>"
        "Our clinical outcomes are above national average in seven of twelve key metrics. "
        "Our patient satisfaction scores improved by 8 percentage points in 2024. However, "
        "our digital infrastructure remains behind peer organisations, nursing vacancy rates "
        "sit at 14.2%, and the Trust carried a £3.2m deficit in the 2023/24 financial year. "
        "Hospital-acquired infection rates at 3.1% exceed the NHS England 2.0% benchmark, "
        "and only 38% of patients rate their overall experience as 'Excellent'."
        "<br/><br/>"
        "This plan directly addresses every one of those gaps whilst consolidating our "
        "existing clinical strengths."
    )
    story.append(Paragraph(context, s["body"]))
    story.append(Spacer(1, 0.7*cm))

    # ── Section 4 — Objectives overview (summary table) ───────────────────
    story.append(Paragraph("4.  Our Five Strategic Objectives", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    ov_header = [
        Paragraph("<b>ID</b>",          ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE)),
        Paragraph("<b>Objective</b>",   ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE)),
        Paragraph("<b>Owner</b>",       ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE)),
    ]
    ov_rows = [ov_header]
    for i, obj in enumerate(objectives):
        colour = OBJ_COLOURS[i]
        ov_rows.append([
            Paragraph(obj["id"], ParagraphStyle("oid", fontName="Helvetica-Bold",
                                                fontSize=8.5, textColor=WHITE,
                                                backColor=colour)),
            Paragraph(obj["title"], ParagraphStyle("ot", fontName="Helvetica", fontSize=8.5,
                                                   textColor=DARK_GREY, leading=12)),
            Paragraph(OBJ_DETAIL[obj["id"]]["owner"].split(",")[0],
                      ParagraphStyle("oo", fontName="Helvetica", fontSize=8, textColor=DARK_GREY, leading=11)),
        ])

    content_w = A4[0] - 3.6*cm
    ov_table = Table(ov_rows, colWidths=[1.4*cm, content_w*0.54, content_w*0.38])
    ov_style = [
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Colour each ID cell
    for i in range(len(objectives)):
        ov_style.append(("BACKGROUND", (0, i+1), (0, i+1), OBJ_COLOURS[i]))
    ov_table.setStyle(TableStyle(ov_style))
    story.append(ov_table)
    story.append(PageBreak())

    # ── Section 5 — One page per objective ────────────────────────────────
    story.append(Paragraph("5.  Strategic Objective Detail Pages", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=4))
    story.append(Paragraph(
        "Each of the following pages provides a complete one-page profile of a single "
        "strategic objective, including the rationale, measurable targets, key risks, "
        "delivery timeline, and the specific actions assigned to deliver it.",
        s["body"]
    ))
    story.append(PageBreak())

    for i, obj in enumerate(objectives):
        detail = OBJ_DETAIL[obj["id"]]
        story += _obj_page(s, obj, detail, i)

    # ── Final sign-off ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Approved by the Board of Directors, Nawaloka Hospital Negombo · January 2025 · {VERSION_SP}",
        s["small_italic"]
    ))

    doc.build(story)
    print(f"✅ Strategic Plan PDF saved → {SP_PDF}")


# ══════════════════════════════════════════════════════════════════════════════
# ACTION PLAN PDF
# ══════════════════════════════════════════════════════════════════════════════

# Maps each action to its primary objective (for the coloured badge)
ACTION_OBJECTIVE_MAP = {
    "A1": "O1", "A2": "O1", "A3": "O1", "A4": "O1",
    "A5": "O2", "A6": "O2", "A7": "O2", "A8": "O2",
    "A9": "O3", "A10": "O3", "A11": "O3", "A12": "O3",
    "A13": "O4", "A14": "O4", "A15": "O4", "A16": "O4",
    "A17": "O5", "A18": "O5", "A19": "O5", "A20": "O5",
}

OBJ_COLOUR_MAP = {
    "O1": OBJ_COLOURS[0],
    "O2": OBJ_COLOURS[1],
    "O3": OBJ_COLOURS[2],
    "O4": OBJ_COLOURS[3],
    "O5": OBJ_COLOURS[4],
}

OBJ_LABEL_MAP = {
    "O1": "Patient Safety",
    "O2": "Technology",
    "O3": "Patient Experience",
    "O4": "Workforce",
    "O5": "Financial Sustainability",
}

# Additional action metadata (owner, timeline, KPI) for realism
ACTION_META = {
    "A1":  ("Director of Infection Prevention",     "Monthly (ongoing)",        "HAI rate < 2%"),
    "A2":  ("Chief Nursing Officer",                "Quarterly workshops",       "Zero medication errors"),
    "A3":  ("Quality & Patient Safety Director",    "12 months to certification","JCI accreditation awarded"),
    "A4":  ("Clinical Governance Lead",             "Q1 2025 — go-live",        "100% incident capture rate"),
    "A5":  ("Chief Information Officer",            "Q3 2025 — full go-live",   "100% digital records"),
    "A6":  ("Head of Digital Operations",           "Q2 2025",                  "30% reduction in no-shows"),
    "A7":  ("IT Security Manager",                  "Q1–Q2 2025",               "Zero critical vulnerabilities"),
    "A8":  ("Medical Director / CIO (joint)",       "Q4 2025 pilot",            "15% faster diagnosis"),
    "A9":  ("Director of Patient Experience",       "Monthly surveys ongoing",  "Satisfaction score > 85%"),
    "A10": ("Head of Outpatient Services",          "Q2 2025",                  "40% reduction in wait time"),
    "A11": ("Director of Patient Experience",       "Q1 2025",                  "12 languages covered"),
    "A12": ("Digital Transformation Lead",          "Q3 2025",                  "60% portal adoption"),
    "A13": ("Director of People & OD",              "12-month rolling cohort",  "30 leaders developed p.a."),
    "A14": ("Head of HR / Wellbeing",               "Q1 2025 — ongoing",        "Burnout rate < 15%"),
    "A15": ("Chief Nursing Officer / HR Director",  "Q1 2025",                  "25% turnover reduction"),
    "A16": ("Director of People & OD",              "Partnership signed Q2 2025","50 interns per year"),
    "A17": ("Chief Financial Officer",              "Annual (Feb each year)",   "10% overhead reduction"),
    "A18": ("Commercial Director",                  "New wing open Q4 2025",    "£2m additional revenue"),
    "A19": ("Chief Financial Officer",              "Rolling submissions",       "£1.5m grant funding p.a."),
    "A20": ("Director of Procurement",              "Q2–Q3 2025",               "15% supply cost reduction"),
}


def build_action_plan_pdf():
    with open(AP_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    actions = data["actions"]

    s = build_styles()
    doc = make_doc(AP_PDF, ACTION_TITLE)
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    intro = [
        "This Operational Action Plan translates the five strategic objectives of the "
        "Nawaloka Hospital Negombo Strategic Plan 2025–2028 into twenty concrete, "
        "time-bound actions. Each action is assigned a named owner, a delivery timeline, "
        "and a measurable key performance indicator.",

        "The plan covers the 2025 financial year and will be refreshed annually in "
        "alignment with the Trust's planning and budget cycle. Progress is reported to "
        "the Executive Management Team monthly and to the Board of Directors quarterly.",

        "Actions are grouped by their primary strategic objective. Where actions support "
        "more than one objective, cross-reference tags are provided.",
    ]
    story += cover_page(s, ACTION_TITLE, VERSION_AP, intro)

    # ── How to read this plan ──────────────────────────────────────────────
    story.append(Paragraph("How to Read This Plan", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    legend_data = [
        [Paragraph("<b>Action ID</b>",  s["body_bold"]),
         Paragraph("Unique reference (A1–A20) used in performance reports and dashboards.", s["body"])],
        [Paragraph("<b>Objective Tag</b>", s["body_bold"]),
         Paragraph("Coloured badge showing the primary strategic objective this action supports.", s["body"])],
        [Paragraph("<b>Owner</b>",      s["body_bold"]),
         Paragraph("Executive or senior leader accountable for delivery.", s["body"])],
        [Paragraph("<b>Timeline</b>",   s["body_bold"]),
         Paragraph("Target completion date or recurring schedule.", s["body"])],
        [Paragraph("<b>KPI</b>",        s["body_bold"]),
         Paragraph("Primary measurable indicator of success.", s["body"])],
    ]
    legend_table = Table(legend_data, colWidths=[3.2*cm, 12.6*cm])
    legend_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT_TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.5, MID_GREY),
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Objective colour key ───────────────────────────────────────────────
    key_cells = []
    for oid, label in OBJ_LABEL_MAP.items():
        colour = OBJ_COLOUR_MAP[oid]
        key_cells.append(
            Table([[Paragraph(f"{oid}  {label}", ParagraphStyle(
                "kp", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE
            ))]],
            colWidths=[3.5*cm])
        )
        key_cells[-1].setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colour),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))

    key_row = Table([key_cells], colWidths=[3.6*cm]*5)
    key_row.setStyle(TableStyle([("LEFTPADDING", (0,0), (-1,-1), 0),
                                  ("RIGHTPADDING", (0,0), (-1,-1), 4)]))
    story.append(key_row)
    story.append(PageBreak())

    # ── Actions grouped by objective ───────────────────────────────────────
    obj_groups = {}
    for action in actions:
        oid = ACTION_OBJECTIVE_MAP.get(action["id"], "O1")
        obj_groups.setdefault(oid, []).append(action)

    obj_titles = {
        "O1": "Strategic Objective O1 — Improve Patient Safety",
        "O2": "Strategic Objective O2 — Modernise Hospital Technology",
        "O3": "Strategic Objective O3 — Enhance Patient Experience and Satisfaction",
        "O4": "Strategic Objective O4 — Develop and Retain a Skilled Workforce",
        "O5": "Strategic Objective O5 — Achieve Financial Sustainability",
    }

    for oid in ["O1", "O2", "O3", "O4", "O5"]:
        colour = OBJ_COLOUR_MAP[oid]
        group_actions = obj_groups.get(oid, [])

        # Section header bar
        sec_data = [[Paragraph(obj_titles[oid], s["obj_title"])]]
        sec_table = Table(sec_data, colWidths=[A4[0] - 3.6*cm])
        sec_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colour),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ]))
        story.append(sec_table)
        story.append(Spacer(1, 0.3*cm))

        for action in group_actions:
            meta = ACTION_META.get(action["id"], ("TBC", "TBC", "TBC"))
            owner, timeline, kpi = meta

            # Action card — left badge + right content
            badge = Table(
                [[Paragraph(action["id"], s["action_id"])]],
                colWidths=[1.5*cm],
                rowHeights=[None]
            )
            badge.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colour),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ]))

            meta_line = (
                f"<b>Owner:</b> {owner}    "
                f"<b>Timeline:</b> {timeline}    "
                f"<b>KPI:</b> {kpi}"
            )
            right_content = [
                Paragraph(action["title"], s["action_title"]),
                Spacer(1, 3),
                Paragraph(action["description"], s["action_body"]),
                Spacer(1, 4),
                Paragraph(meta_line, s["small_italic"]),
            ]
            right_col = Table([[rc] for rc in right_content],
                              colWidths=[A4[0] - 3.6*cm - 1.5*cm - 0.3*cm])
            right_col.setStyle(TableStyle([
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ]))

            card_data = [[badge, right_col]]
            card = Table(card_data,
                         colWidths=[1.5*cm, A4[0] - 3.6*cm - 1.5*cm - 0.3*cm],
                         hAlign="LEFT")
            card.setStyle(TableStyle([
                ("BACKGROUND",    (1, 0), (1, 0), LIGHT_GREY),
                ("TOPPADDING",    (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING",   (1, 0), (1, 0), 12),
                ("RIGHTPADDING",  (1, 0), (1, 0), 12),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW",     (0, 0), (-1, -1), 1.5, colour),
            ]))

            story.append(KeepTogether([card, Spacer(1, 0.25*cm)]))

        story.append(Spacer(1, 0.4*cm))
        story.append(PageBreak())

    # ── Sign-off page ──────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Document Approval and Sign-off", s["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=12))

    signoff_data = [
        [Paragraph("<b>Role</b>",            s["body_bold"]),
         Paragraph("<b>Name</b>",            s["body_bold"]),
         Paragraph("<b>Signature</b>",       s["body_bold"]),
         Paragraph("<b>Date</b>",            s["body_bold"])],
        ["Chief Executive Officer",      "Dr. Patricia Okonkwo",   "_________________", "Jan 2025"],
        ["Medical Director",             "Mr. James Acheampong",   "_________________", "Jan 2025"],
        ["Chief Nursing Officer",        "Ms. Sunita Patel",       "_________________", "Jan 2025"],
        ["Chief Financial Officer",      "Mr. David Reeves",       "_________________", "Feb 2025"],
        ["Chief Information Officer",    "Ms. Leila Moussaoui",    "_________________", "Feb 2025"],
        ["Director of People & OD",      "Ms. Ama Forson",         "_________________", "Feb 2025"],
    ]
    signoff_table = Table(signoff_data, colWidths=[5*cm, 4.5*cm, 4*cm, 2.3*cm])
    signoff_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.5, MID_GREY),
    ]))
    story.append(signoff_table)

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Approved by the Board of Directors, Nawaloka Hospital Negombo · February 2025 · {VERSION_AP}",
        s["small_italic"]
    ))

    doc.build(story)
    print(f"✅ Action Plan PDF saved → {AP_PDF}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("📄 Generating hospital PDF documents...")
    build_strategic_plan_pdf()
    build_action_plan_pdf()
    print("🎉 Both PDFs generated successfully in data/")
