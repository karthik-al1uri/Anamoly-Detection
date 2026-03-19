def query_index(query: str) -> dict[str, str]:
    return {
        'query': query,
        'match': 'Placeholder SOP match',
    }


if __name__ == '__main__':
    print(query_index('deep scratch on aluminum casing'))
