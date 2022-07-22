"""
Created by khadm on 1/5/2022
Feature: 
"""


def test_loop(test_cases: list, test_name: str = 'general _test', logger=None):
    success = True

    for test_case in test_cases:
        try:
            assert test_case['result'] == test_case['expected']
        except:
            success = False
            logger.error(f'{test_case["name"]}: {test_case["error_message"]}')

    if success is True:
        logger.debug(f'{test_name}:  All tests passed')
    return success
