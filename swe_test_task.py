from composer_agent import TrialAgent
import pathlib
import json
import git
import subprocess
import os
agent = TrialAgent()

print(agent.show_available_tools())

class SWE_Task:
    def __init__(self, task_id):
        self.task_id = task_id
        self.task_data = self._load_task_data(task_id)
        self.repo_path = self.task_data['repo']
        self.base_commit = self.task_data['base_commit']
        self.patch = self.task_data['patch']
        self.test_patch = self.task_data['test_patch']
        self.problem_statement = self.task_data['problem_statement']
        self.hints_text = self.task_data['hints_text']
        self.repo = None
        self.PASS_TO_PASS = self.task_data['PASS_TO_PASS']
        self.FAIL_TO_PASS = self.task_data['FAIL_TO_PASS']

    def _load_task_data(self, task_id):
        try:
            with open('data.json', 'r') as f:
                task_data = json.load(f)
                return task_data[task_id]
        except Exception as e:
            print(f"Error loading task data: {str(e)}")
            return None
        
    def _set_env(self):
        try:
            # Extract repo name from the format "owner/repo"
            local_dir = os.getcwd()
            repo_name = self.repo_path.split('/')[-1]
            if not pathlib.Path(repo_name).exists():
                repo_url = f"https://github.com/{self.repo_path}.git"
                git.Repo.clone_from(repo_url, repo_name)
                print(f"Cloned repository: {self.repo_path}")
            else:
                print(f"Repository {repo_name} already exists")
            
            self.repo = git.Repo(repo_name)
            current_commit = self.repo.head.commit.hexsha
            
            if current_commit != self.base_commit:
                self.repo.git.checkout(self.base_commit)
                print(f"Checked out base commit: {self.base_commit}")
            else:
                print(f"Already on base commit: {self.base_commit}")
                
        except Exception as e:
            print(f"Error setting up environment: {str(e)}")
            return None
        
        return repo_name

    def _apply_patch(self, repo_name):
        try:
            os.chdir(repo_name)
            print(f"Moved to repo and applying patch at {os.getcwd()}")
            self.repo.git.apply(self.test_patch)
            print(f"Applied patch: {self.test_patch}")
        except Exception as e:
            print(f"Error applying patch: {str(e)}")
            return None
        
    def _run_tests(self, test_names):
        try:
            cmd = ["python", "manage.py", "test"] + test_names
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        except Exception as e:
            print(f"Error running tests: {str(e)}")
            return None
        
    def _check_test_results(self, stdout, test_names, expect_pass=True):
        results = {}
        for test in test_names:
            test_name = test.split(".")[-1]  # e.g., "test_order_by_multiline_sql"
            if expect_pass:
                passed = f"OK" in stdout and test_name in stdout and "FAILED" not in stdout
            else:
                passed = f"FAILED" in stdout and test_name in stdout
            results[test] = "PASSED" if passed else "FAILED"
        return results
    
    def invoke(self, test_names='expressions.tests', expect_pass=True):
        repo_name = self._set_env()
        current_cwd = os.getcwd()
        root_dir = os.path.join(current_cwd, "django")
        # Change directory to the Django project root
        os.chdir(root_dir)
        print(f"Changed directory to {os.getcwd()}")

        patch_file = "patch.diff"

        # Run the patch command using subprocess
        # subprocess.run(["git", "apply", patch_file], check=True)

        # Run the tests using pytest
        result = subprocess.run(
            ["pytest", "-k", test_names],
            check=False,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(result.stderr)

    def write_patch(self):
        patch_file = "patch.diff"
        content = "diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py\n--- a/tests/expressions/tests.py\n+++ b/tests/expressions/tests.py\n@@ -384,6 +384,29 @@ def test_order_by_exists(self):\n         )\n         self.assertSequenceEqual(mustermanns_by_seniority, [self.max, mary])\n \n+    def test_order_by_multiline_sql(self):\n+        raw_order_by = (\n+            RawSQL('''\n+                CASE WHEN num_employees > 1000\n+                     THEN num_chairs\n+                     ELSE 0 END\n+            ''', []).desc(),\n+            RawSQL('''\n+                CASE WHEN num_chairs > 1\n+                     THEN 1\n+                     ELSE 0 END\n+            ''', []).asc()\n+        )\n+        for qs in (\n+            Company.objects.all(),\n+            Company.objects.distinct(),\n+        ):\n+            with self.subTest(qs=qs):\n+                self.assertSequenceEqual(\n+                    qs.order_by(*raw_order_by),\n+                    [self.example_inc, self.gmbh, self.foobar_ltd],\n+                )\n+\n     def test_outerref(self):\n         inner = Company.objects.filter(point_of_contact=OuterRef('pk'))\n         msg = (\n)"
        with open(patch_file, "w") as f:
            f.write(content)
        print(f"Wrote patch to {patch_file}")

task = SWE_Task(1)

print(task.invoke())



