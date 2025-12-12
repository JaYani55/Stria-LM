#!/usr/bin/env python3
"""
Stria-LM Export Script
Export projects from the current database to portable SQLite files.

Usage:
    python scripts/export_to_sqlite.py <project_name> [output_path]
    python scripts/export_to_sqlite.py --all [output_dir]
    
Examples:
    python scripts/export_to_sqlite.py my_project
    python scripts/export_to_sqlite.py my_project exports/my_project.db
    python scripts/export_to_sqlite.py --all exports/
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATABASE_TYPE, PROJECTS_DIR
from src.database import get_database


def export_project(project_name: str, output_path: str = None) -> bool:
    """
    Export a single project to a SQLite database file.
    
    Args:
        project_name: Name of the project to export
        output_path: Output path for the .db file (default: exports/{project_name}/{project_name}.db)
        
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = f"exports/{project_name}/{project_name}.db"
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting project '{project_name}' to {output_path}...")
    
    try:
        db = get_database()
        
        # Check if project exists
        if not db.get_project(project_name):
            print(f"Error: Project '{project_name}' not found.")
            return False
        
        # Export
        db.export_to_sqlite(project_name, output_path)
        
        print(f"✓ Successfully exported '{project_name}' to {output_path}")
        return True
        
    except NotImplementedError:
        if DATABASE_TYPE == "sqlite":
            print(f"Note: Project is already SQLite. Copying database file...")
            
            # For SQLite, just copy the file
            import shutil
            src_path = PROJECTS_DIR / project_name / f"{project_name}.db"
            
            if src_path.exists():
                shutil.copy2(src_path, output_path)
                print(f"✓ Copied {src_path} to {output_path}")
                return True
            else:
                print(f"Error: Source database not found at {src_path}")
                return False
        else:
            print(f"Error: Export not supported for current database type.")
            return False
            
    except Exception as e:
        print(f"Error exporting project: {e}")
        return False


def export_all_projects(output_dir: str = "exports") -> int:
    """
    Export all projects to SQLite database files.
    
    Args:
        output_dir: Directory to store exported files
        
    Returns:
        Number of successfully exported projects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting all projects to {output_dir}/...")
    print(f"Current database type: {DATABASE_TYPE}")
    print()
    
    try:
        db = get_database()
        projects = db.list_projects()
        
        if not projects:
            print("No projects found.")
            return 0
        
        print(f"Found {len(projects)} project(s) to export.")
        print()
        
        success_count = 0
        for project_name in projects:
            project_output = output_path / project_name / f"{project_name}.db"
            if export_project(project_name, str(project_output)):
                success_count += 1
            print()
        
        print(f"Export complete: {success_count}/{len(projects)} projects exported successfully.")
        return success_count
        
    except Exception as e:
        print(f"Error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Export Stria-LM projects to portable SQLite database files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export a single project:
    python scripts/export_to_sqlite.py my_project
    
  Export with custom path:
    python scripts/export_to_sqlite.py my_project ~/backups/my_project.db
    
  Export all projects:
    python scripts/export_to_sqlite.py --all
    python scripts/export_to_sqlite.py --all exports/
        """
    )
    
    parser.add_argument(
        "project_name",
        nargs="?",
        help="Name of the project to export"
    )
    
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Output path for the .db file or directory (for --all)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all projects"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available projects"
    )
    
    args = parser.parse_args()
    
    # List projects
    if args.list:
        try:
            db = get_database()
            projects = db.list_projects()
            
            if not projects:
                print("No projects found.")
            else:
                print(f"Available projects ({DATABASE_TYPE}):")
                for project in projects:
                    print(f"  - {project}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # Export all projects
    if args.all:
        output_dir = args.project_name or args.output_path or "exports"
        success = export_all_projects(output_dir)
        sys.exit(0 if success > 0 else 1)
    
    # Export single project
    if not args.project_name:
        parser.print_help()
        print("\nError: Please specify a project name or use --all")
        sys.exit(1)
    
    success = export_project(args.project_name, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
