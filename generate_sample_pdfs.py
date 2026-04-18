#!/usr/bin/env python3
"""Generate sample HR PDF documents for DayOne AI demo organizations."""

from pathlib import Path
import sys
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
except ImportError:
    print("reportlab not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

DATA_DIR = Path(__file__).parent / "data"

def create_acme_handbook_pdf():
    """Create Acme Corp Benefits Guide PDF."""
    pdf_path = DATA_DIR / "org_acme" / "benefits_guide.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#0066cc',
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    story.append(Paragraph("ACME CORPORATION<br/>Benefits Guide 2025", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Content
    content = [
        ("Health Insurance", 
         "ACME Corporation offers comprehensive health insurance coverage starting on your first day of employment. "
         "Our plans include medical, dental, and vision coverage. Employee contribution is 15% of premium, with the company covering 85%."),
        
        ("Retirement Planning",
         "Employees are eligible for our 401(k) plan after 30 days of employment. ACME matches 100% of contributions up to 3% of salary. "
         "Vesting schedule: 20% per year, fully vested after 5 years."),
        
        ("Paid Time Off",
         "New employees receive 15 days of PTO annually, increasing to 20 days after 3 years and 25 days after 5 years. "
         "PTO can be carried over but maximum carryover is 5 days per year."),
        
        ("Professional Development",
         "ACME provides $2,500 annually for professional development and training. This includes tuition reimbursement, certifications, and conferences. "
         "Approval required from your manager."),
        
        ("Work from Home",
         "Eligible employees can work remotely up to 3 days per week. Core hours are 10 AM - 3 PM EST. "
         "Additional remote arrangements can be negotiated with your manager."),
    ]
    
    for title, text in content:
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#0066cc',
            spaceAfter=12,
        )
        story.append(Paragraph(title, heading_style))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
    
    doc.build(story)
    print(f"Created {pdf_path}")

def create_globex_handbook_pdf():
    """Create Globex Corp Employee Handbook PDF."""
    pdf_path = DATA_DIR / "org_globex" / "employee_handbook_extended.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#cc6600',
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    story.append(Paragraph("GLOBEX CORPORATION<br/>Employee Handbook", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Content
    content = [
        ("Company Values",
         "At Globex, we believe in innovation, integrity, and collaboration. Every employee is expected to uphold these values "
         "in their daily interactions and decision-making."),
        
        ("Code of Conduct",
         "All employees must adhere to our Code of Conduct, which includes: respect for colleagues, confidentiality of company information, "
         "ethical business practices, and zero tolerance for harassment or discrimination."),
        
        ("Performance Reviews",
         "Annual performance reviews are conducted in December. Mid-year check-ins occur in June. Employees are evaluated on competencies, "
         "goals achievement, and compliance with company policies."),
        
        ("Dress Code",
         "Globex maintains a business casual dress code. Jeans are permitted on Fridays. Avoid overly casual or revealing clothing. "
         "Client-facing meetings require business formal attire."),
        
        ("Communication Policy",
         "Company email and collaboration tools are for business use. Personal use should be minimal and appropriate. "
         "The company reserves the right to monitor business communications."),
        
        ("Safety & Security",
         "All employees must complete safety training within 30 days of hire. Report any safety concerns immediately to HR. "
         "Badges must be worn at all times in restricted areas."),
    ]
    
    for title, text in content:
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#cc6600',
            spaceAfter=12,
        )
        story.append(Paragraph(title, heading_style))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
    
    doc.build(story)
    print(f"Created {pdf_path}")

def create_acme_policies_pdf():
    """Create Acme Corp Policies PDF."""
    pdf_path = DATA_DIR / "org_acme" / "company_policies.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#0066cc',
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    story.append(Paragraph("ACME CORPORATION<br/>Company Policies", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    content = [
        ("Attendance Policy",
         "Employees are expected to arrive on time and notify their manager at least 2 hours before an absence. "
         "Three unexcused absences in 90 days may result in disciplinary action."),
        
        ("Innovation Program",
         "ACME encourages employee innovation. Submit ideas for process improvements through the Innovation Portal. "
         "Selected ideas receive recognition and potential implementation bonuses."),
        
        ("Wellness Program",
         "Free gym membership, mental health counseling, and wellness classes are available to all employees. "
         "Participate in our quarterly wellness challenges for incentives."),
        
        ("Mentorship",
         "All new hires are assigned a mentor in their first week. Formal mentorship program runs for 12 weeks. "
         "Mentors receive stipend bonuses and professional development credits."),
    ]
    
    for title, text in content:
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#0066cc',
            spaceAfter=12,
        )
        story.append(Paragraph(title, heading_style))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
    
    doc.build(story)
    print(f"Created {pdf_path}")

def create_globex_career_pdf():
    """Create Globex Corp Career Development PDF."""
    pdf_path = DATA_DIR / "org_globex" / "career_development.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#cc6600',
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    story.append(Paragraph("GLOBEX CORPORATION<br/>Career Development", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    content = [
        ("Career Paths",
         "Globex offers multiple career paths: Individual Contributor Track, Management Track, and Specialist Track. "
         "Employees can switch paths with manager approval."),
        
        ("Promotion Criteria",
         "Promotions are based on performance, tenure (minimum 1 year in current role), and skill development. "
         "Apply through the Careers Portal or discuss with your manager."),
        
        ("Training Programs",
         "Globex provides: Leadership Development Program (for managers), Technical Skills Bootcamp, and Executive Coaching. "
         "Employees are encouraged to pursue 2-3 trainings annually."),
        
        ("Tuition Reimbursement",
         "Full reimbursement up to $5,000 annually for degree programs or professional certifications. "
         "Requires pre-approval and a 1-year retention agreement post-completion."),
        
        ("Internal Job Postings",
         "All positions are posted internally first. Current employees have 5 days before external posting. "
         "Internal transfers are encouraged to reduce external hiring."),
    ]
    
    for title, text in content:
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#cc6600',
            spaceAfter=12,
        )
        story.append(Paragraph(title, heading_style))
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
    
    doc.build(story)
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    print("Generating sample HR PDF documents...\n")
    create_acme_handbook_pdf()
    create_acme_policies_pdf()
    create_globex_handbook_pdf()
    create_globex_career_pdf()
    print("\nAll sample PDFs created successfully!")
