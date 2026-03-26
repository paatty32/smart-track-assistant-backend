import argparse

from agents.mcp_agents import AGENT_FACTORY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent',
                        choices=AGENT_FACTORY.keys(),
                        help="Welchen Agenten starten"
                        )
    return parser.parse_args()

