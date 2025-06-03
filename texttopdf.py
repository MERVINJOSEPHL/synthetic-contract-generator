from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def text_file_to_pdf(input_file_path, output_pdf_path):
    """
    Convert content from a text file to a PDF file using ReportLab.
    
    Args:
        input_file_path (str): Path to the input text file (e.g., containing Python code).
        output_pdf_path (str): Path for the output PDF file.
    """
    try:
        # Read the content from the text file
        with open(input_file_path, "r", encoding="utf-8") as file:
            input_text = file.read()
        
        # Create a PDF document
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Create a custom style for code (monospaced font, preserving formatting)
        code_style = ParagraphStyle(
            name='Code',
            fontName='Courier',  # Monospaced font for code
            fontSize=10,
            leading=12,  # Line spacing
            spaceBefore=0,
            spaceAfter=0,
            wordWrap=None,  # Prevent word wrapping to maintain code structure
            textColor=colors.black
        )
        
        # Initialize the story (content) for the PDF
        story = []
        
        # Split the input text into lines
        lines = input_text.splitlines()
        
        # Process each line to handle basic formatting
        for line in lines:
            # Escape HTML characters to prevent rendering issues
            line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Replace tabs with spaces for consistent rendering
            line = line.replace('\t', '    ')
            
            # Handle empty lines
            if not line.strip():
                story.append(Spacer(1, 12))  # Add vertical space for empty lines
                continue
            
            # Wrap the line in a Paragraph with the code style
            story.append(Paragraph(line, code_style))
        
        # Build the PDF
        doc.build(story)
        print(f"PDF successfully created at: {output_pdf_path}")
    
    except FileNotFoundError:
        print(f"Error: The input file '{input_file_path}' was not found. Please check the file path.")
    except Exception as e:
        print(f"Error converting text file to PDF: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Specify the input text file path (replace with your actual file path)
    input_file_path = "sample.txt"  # Adjust this to your text file's name/path
    
    # Define output PDF path
    output_pdf_path = "synthetic_contract_generator.pdf"
    
    # Convert the text file to PDF
    text_file_to_pdf(input_file_path, output_pdf_path)