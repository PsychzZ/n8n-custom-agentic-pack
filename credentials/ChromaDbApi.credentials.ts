import { ICredentialType, INodeProperties } from 'n8n-workflow';

/**
 * @description n8n credential type for ChromaDB API access. Stores the host URL
 * and an optional bearer token for authentication. Used by VectorStoreNode when
 * ChromaDB is deployed with auth enabled.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
 * Secure, configurable access to the document store is a prerequisite for
 * production RAG deployments. Credential management via n8n prevents hardcoding
 * of sensitive access tokens in workflow definitions.
 *
 * @thesis_note DE: ChromaDbApiCredentials kapselt die Verbindungskonfiguration
 * für ChromaDB. Die Auslagerung in einen separaten Credential-Typ folgt dem
 * n8n-Sicherheitsmodell und ermöglicht eine sichere Weitergabe von Workflows
 * ohne Preisgabe von Zugangsdaten.
 */
export class ChromaDbApi implements ICredentialType {
  name = 'chromaDbApi';
  displayName = 'ChromaDB API';
  documentationUrl = 'https://docs.trychroma.com/';

  properties: INodeProperties[] = [
    {
      displayName: 'ChromaDB URL',
      name: 'chromaUrl',
      type: 'string',
      default: 'http://localhost:8000',
      placeholder: 'http://localhost:8000',
      required: true,
      description: 'The base URL of your ChromaDB instance',
    },
    {
      displayName: 'Bearer Token',
      name: 'bearerToken',
      type: 'string',
      typeOptions: { password: true },
      default: '',
      description:
        'Optional bearer token for ChromaDB deployments with token-based authentication',
    },
    {
      displayName: 'Tenant',
      name: 'tenant',
      type: 'string',
      default: 'default_tenant',
      description: 'ChromaDB tenant name (default: default_tenant)',
    },
    {
      displayName: 'Database',
      name: 'database',
      type: 'string',
      default: 'default_database',
      description: 'ChromaDB database name (default: default_database)',
    },
  ];
}
